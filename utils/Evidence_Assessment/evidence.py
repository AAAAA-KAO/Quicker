from typing import List, Union, Dict
import json, os
import pandas as pd
import logging
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from utils.Evidence_Assessment.outcome import Outcome
from utils.Evidence_Assessment.paper import Paper, StudyDesign
from utils.Evidence_Assessment.prompt import (
    paper_analysis_prompttemplate,
    STUDY_DESIGN_QUESTION,
    study_design_json_parser,
)
from utils.Evidence_Assessment.rag import analyze_paper_by_rag
from utils.Evidence_Assessment.grade import GRADEAssessment


class Evidence:
    def __init__(
        self,
        pico_idx: str,
        disease: str,  #! 之后改叫topic吧
        clinical_question: str,
        intervention: str,
        comparator: str,
        outcome_list: List[Outcome],
        paperinfo_list: List[Paper],
        # outcomeinfo_path: str,
        # paperinfo_path: str,
        embeddings,
        model,
        additional_requirements: dict = None,
        comparator_postfix: str = None,
        annotation: Dict[str, dict] = None,  # outcome: {annotation}
    ):
        self.pico_idx = pico_idx
        self.disease = disease
        self.clinical_question = clinical_question
        self.intervention = intervention
        self.comparator = comparator
        self.outcome_list = outcome_list
        self.paper_list = paperinfo_list
        self.embeddings = embeddings
        self.model = model
        self.additional_requirements = (
            additional_requirements if additional_requirements else {}
        )
        logging.debug(
            'additional_requirements for evidence assessment: {}'.format(
                additional_requirements
            )
        )
        self.comparator_postfix = comparator_postfix

        # # load outcomeinfo JSON
        # with open(outcomeinfo_path, "r") as f:
        #     self.outcomeinfo = json.load(
        #         f
        #     )  # this outcomeinfo contains all the outcomes of this pico
        # self.outcome_list = [Outcome.from_dict(outcome) for outcome in self.outcomeinfo]

        # # load paperinfo JSON
        # with open(paperinfo_path, "r") as f:
        #     self.paperinfo = json.load(
        #         f
        #     )  # this paperinfo contains all the papers of this pico

        # self.paper_list = [Paper.from_dict(paper) for paper in self.paperinfo]
        # choose the design to be assessed
        # if self.additional_requirements.get('design_to_be_assessed') is None:
        self.design_function_map = {
            StudyDesign.SYSTEMATIC_REVIEW: self.assess_systematic_review,
            StudyDesign.RANDOMIZED_CONTROLLED_TRIAL: self.assess_rct,
            StudyDesign.COHORT_STUDY: self.assess_cohort_study,
            StudyDesign.META_ANALYSIS: self.assess_meta_analysis,
            StudyDesign.OTHER_OBSERVATIONAL_STUDY: self.assess_cohort_study,
        }
        # else:
        #     self.design_function_map = {}
        #     for i in self.additional_requirements['design_to_be_assessed']:
        #         if i == 'SYSTEMATIC_REVIEW':
        #             self.design_function_map[StudyDesign.SYSTEMATIC_REVIEW] = (
        #                 self.assess_systematic_review
        #             )
        #         elif i == 'RANDOMIZED_CONTROLLED_TRIAL':
        #             self.design_function_map[
        #                 StudyDesign.RANDOMIZED_CONTROLLED_TRIAL
        #             ] = self.assess_rct
        #         elif i == 'COHORT_STUDY':
        #             self.design_function_map[StudyDesign.COHORT_STUDY] = (
        #                 self.assess_cohort_study
        #             )
        #         elif i == 'META_ANALYSIS':
        #             self.design_function_map[StudyDesign.META_ANALYSIS] = (
        #                 self.assess_meta_analysis
        #             )
        logging.debug(
            'design_function_map for evidence assessment: {}'.format(
                self.design_function_map
            )
        )

        self.annotation = annotation if annotation else {}

    def __str__(self):
        all_outcome_str = ""
        for outcome in self.outcome_list:
            all_outcome_str += str(outcome) + "\n"
        return f"Clinical Question: {self.clinical_question} \n" + all_outcome_str

    def add_outcome(self, outcome: Outcome):
        self.outcome_list.append(outcome)

    def assess_outcome(self, outcome: Outcome):
        '''
        评估单个outcome，根据相关paper的研究设计，进行不同的评估，最后融合出该outcome的综合评估结果
        '''
        # get the papers related to this outcome
        related_paper = [
            p
            for p in self.paper_list
            if p.paper_uid in outcome.related_paper_list  # 在这已经选出了关联的paper
        ]
        logging.info(
            f'Found {len(related_paper)} related papers for outcome {outcome.outcome_uid}'
        )

        # check if the related papers have characteristics and group the papers by study design
        study_design_group = {}
        for paper in related_paper:
            if not paper.study_design or not paper.characteristics:
                logging.info(
                    f'updating study design and characteristics of paper {paper.paper_uid}'
                )
                paper.update_study_design_and_characteristics(
                    **self.analyze_paper(
                        paper=paper,
                        embeddings=self.embeddings,
                        model=self.model,
                        disease=self.disease,
                        method='RAG',  #! 这里应该是一个参数
                    )
                )
            if paper.study_design not in study_design_group:
                study_design_group[paper.study_design] = []
            study_design_group[paper.study_design].append(
                paper
            )  # group the papers by study design
        logging.info(
            f'Grouped the papers by study design for outcome {outcome.outcome_uid}'
        )
        logging.info(f'Study design group: {study_design_group.keys()}')
        # assess the outcome
        self.assess_outcome_by_study_design(outcome, study_design_group)
        return outcome

    @staticmethod
    def analyze_paper(
        paper: Paper,
        embeddings,
        model,
        disease: str,
        method: str = 'RAG',
        given_option: dict = {},  #! 提供给定component的选项，便于控制答案
        reupdate_component_list: list = [],
    ) -> dict:
        '''
        This function is used to analyze the paper and extract the characteristics.
        Specifically, it should use the model to extract the study design and characteristics from the paper and update the paper object.
        The characteristics should be stored in the paper object as a dictionary.

        Args:
            paper: the paper object to be analyzed
            embeddings: the embeddings used to analyze the paper
            model: the model object used to analyze the paper
            method: the method used to analyze the paper, default is 'RAG'
            disease: the disease of the paper
            given_option: the given options for the component extraction, default is {}

        Returns:
            A dictionary containing the study design and characteristics extracted from the paper.
        '''

        if method == 'RAG':
            # check if the vector database is existed
            vectorstore = paper.get_vector_store(embeddings=embeddings, model=model)

            study_design_retriever = vectorstore.as_retriever(
                search_type="similarity", search_kwargs={"k": 5}
            )

            study_design_prompttemplate = paper_analysis_prompttemplate.partial(
                disease=disease,
                format_instructions=study_design_json_parser.get_format_instructions(),
            )
            logging.info("RAG analysis for study design of paper")

            study_design_analysis = analyze_paper_by_rag(
                study_design_retriever,
                model,
                study_design_prompttemplate,
                study_design_json_parser,
                STUDY_DESIGN_QUESTION,
            )

            study_design = study_design_analysis.study_design

            logging.info(f"Paper id: {paper.paper_uid} |  study design: {study_design}")
            logging.info(
                f"Paper id: {paper.paper_uid} |  study design related content: {study_design_analysis.related_content}"
            )

            characteristics = dict(
                study_design_related_content=study_design_analysis.related_content
            )
            component_list = ['population', 'intervention', 'comparator', 'outcome']
            if reupdate_component_list:
                if paper.characteristics:
                    characteristics.update(paper.characteristics)
                component_list = reupdate_component_list

            if study_design == StudyDesign.NOT_APPLICABLE:
                return {
                    "study_design": study_design,
                    "characteristics": characteristics,
                }

            #     logging.error("RAG analysis for SR paper has not been implemented yet")
            # elif study_design == StudyDesign.RANDOMIZED_CONTROLLED_TRIAL:
            logging.info(f"Paper id: {paper.paper_uid} | Analyze characteristic by RAG")

            # find the PICO elements in the paper
            top_k = os.getenv('TOP_K', 5)
            pico_retriever = vectorstore.as_retriever(
                search_type="similarity", search_kwargs={"k": int(top_k)}
            )

            from utils.Evidence_Assessment.rag import extract_pico_from_paper
            from langchain_core.runnables import RunnableLambda

            def flatten_outcome_options(outcome_options):
                if isinstance(outcome_options, str):
                    return [outcome_options]
                if isinstance(outcome_options, list):
                    flattened = []
                    for option in outcome_options:
                        flattened.extend(flatten_outcome_options(option))
                    return flattened
                return []

            pre_outcome_component_list = [
                component for component in component_list if component != 'outcome'
            ]
            extracted_component_list = RunnableLambda(
                lambda component: extract_pico_from_paper(
                    component=component,
                    given_option=given_option,
                    disease=disease,
                    pico_retriever=pico_retriever,
                    model=model,
                    abstract=paper.abstract,
                )
            ).batch(pre_outcome_component_list)

            for component in extracted_component_list:
                characteristics.update(component)

            if 'outcome' in component_list:
                outcome_given_option = dict(given_option)
                outcome_options = outcome_given_option.get('outcome')
                if isinstance(outcome_options, dict):
                    matched_comparators = (
                        characteristics.get('comparator', {}).get('comparator', [])
                    )
                    if isinstance(matched_comparators, str):
                        matched_comparators = [matched_comparators]

                    filtered_outcomes = []
                    for comparator in matched_comparators:
                        if comparator in outcome_options:
                            filtered_outcomes.extend(
                                flatten_outcome_options(outcome_options[comparator])
                            )

                    if not filtered_outcomes:
                        filtered_outcomes = flatten_outcome_options(
                            list(outcome_options.values())
                        )

                    outcome_given_option['outcome'] = filtered_outcomes

                characteristics.update(
                    extract_pico_from_paper(
                        component='outcome',
                        given_option=outcome_given_option,
                        disease=disease,
                        pico_retriever=pico_retriever,
                        model=model,
                        abstract=paper.abstract,
                    )
                )

            logging.info(
                f"Paper id: {paper.paper_uid} |   Extracted characteristics: {characteristics}"
            )

            # elif study_design == StudyDesign.COHORT_STUDY:
            #     logging.error(
            #         "RAG analysis for cohort study paper has not been implemented yet"
            #     )
            # elif study_design == StudyDesign.META_ANALYSIS:
            #     logging.error(
            #         "RAG analysis for meta analysis paper has not been implemented yet"
            #     )

        else:
            raise NotImplementedError(f'The method {method} is not implemented')

        return {
            "study_design": study_design,
            "characteristics": characteristics,
        }

    def assess_outcome_by_study_design(
        self, outcome: Outcome, study_design_group: dict
    ):
        '''
        This function is used to assess the outcome based on the study design group.
        Specifically, it should use the study design group to assess the outcome and update the state of the outcome.
        '''
        for study_design, paper_list in study_design_group.items():
            self.design_function_map[study_design](outcome, paper_list)

    def assess_systematic_review(self, outcome: Outcome, paper_list: List[Paper]):
        '''
        This function is used to assess the outcome based on the systematic review papers.
        Specifically, it should use the systematic review papers to assess the outcome.
        '''
        logging.warning(
            'Systematic review assessment has not been implemented yet; '
            'skipping %s systematic review paper(s).',
            len(paper_list),
        )
        outcome.update_assessment_results(
            assessment_name='Systematic Review',
            sub_assessment_name='implementation_status',
            sub_assessment_result=(
                'Systematic review assessment has not been implemented yet; '
                'this study design group was skipped.'
            ),
        )
        pass

    def assess_rct(self, outcome: Outcome, paper_list: List[Paper]):
        '''
        This function is used to assess the outcome based on the randomized controlled trial papers.
        Specifically, it should use the randomized controlled trial papers to assess the outcome.
        '''

        # ROB2 assessment

        # GRADE assessment
        logging.info(f'Starting to assess the outcome {outcome.outcome_uid} by GRADE')
        GRADEAssessment(
            outcome=outcome,
            paper_list=paper_list,
            disease=self.disease,
            embeddings=self.embeddings,
            model=self.model,
            additional_requirements_for_GRADE=self.additional_requirements.get(
                "GRADE", {}
            ),
            annotation=self.annotation.get(outcome.outcome, {}),
        ).run_assessment()

    def assess_cohort_study(self, outcome: Outcome, paper_list: List[Paper]):
        '''
        This function is used to assess the outcome based on the cohort study papers.
        Specifically, it should use the cohort study papers to assess the outcome.
        '''
        logging.warning(
            'Cohort study assessment has not been implemented yet; '
            'skipping %s cohort/observational paper(s).',
            len(paper_list),
        )
        outcome.update_assessment_results(
            assessment_name='Cohort Study',
            sub_assessment_name='implementation_status',
            sub_assessment_result=(
                'Cohort/observational study assessment has not been implemented yet; '
                'this study design group was skipped.'
            ),
        )
        pass

    def assess_meta_analysis(self, outcome: Outcome, paper_list: List[Paper]):
        '''
        This function is used to assess the outcome based on the meta analysis papers.
        Specifically, it should use the meta analysis papers to assess the outcome.
        '''
        logging.warning(
            'Meta analysis assessment has not been implemented yet; '
            'skipping %s meta-analysis paper(s).',
            len(paper_list),
        )
        outcome.update_assessment_results(
            assessment_name='Meta-Analysis',
            sub_assessment_name='implementation_status',
            sub_assessment_result=(
                'Meta-analysis assessment has not been implemented yet; '
                'this study design group was skipped.'
            ),
        )
        pass

    def assess_evidence(self):
        '''
        评估所有的outcome，形成综合所有outcome的评估结果
        '''
        # assess all the outcomes one by one
        logging.info('Starting to assess the whole evidence')
        for outcome in self.outcome_list:
            logging.info(f'Starting to assess the outcome: {outcome}')
            self.assess_outcome(outcome)

        # outcome_assessment_chain = RunnableLambda(
        #     lambda outcome: self.assess_outcome(outcome)
        # )

        # _ = outcome_assessment_chain.batch(self.outcome_list)
        # todo: overall certainty assessment

        pass
