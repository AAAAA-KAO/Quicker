import logging
import os
from PyPaperBot.__main__ import start as start_download_pdf
from PyPaperBot.proxy import proxy as proxy_func


def download_pdf_by_paperbot(
    dwn_dir: str,
    query: str = None,
    doi_file: str = None,
    doi: str = None,
    cites: str = None,
    scholar_results: int = 1,
    scholar_pages: str = '1',
    proxy: list = [],
    min_year: int = None,
    max_dwn_year: int = None,
    max_dwn_cites: int = None,
    journal_filter: str = None,
    restrict: int = None,
    scihub_mirror: str = None,
    selenium_chrome_version: int = None,
    use_doi_as_filename: bool = False,
    annas_archive_mirror: str = None,
    skip_words: str = None,
    single_proxy: str = None,
):
    # parser = argparse.ArgumentParser(
    #     description='PyPaperBot is python tool to search and dwonload scientific papers using Google Scholar, Crossref and SciHub')
    # parser.add_argument('--query', type=str, default=None,
    #                     help='Query to make on Google Scholar or Google Scholar page link')
    # parser.add_argument('--skip-words', type=str, default=None,
    #                     help='List of comma separated works. Papers from Scholar containing this words on title or summary will be skipped')
    # parser.add_argument('--cites', type=str, default=None,
    #                     help='Paper ID (from scholar address bar when you search citations) if you want get only citations of that paper')
    # parser.add_argument('--doi', type=str, default=None,
    #                     help='DOI of the paper to download (this option uses only SciHub to download)')
    # parser.add_argument('--doi-file', type=str, default=None,
    #                     help='File .txt containing the list of paper\'s DOIs to download')
    # parser.add_argument('--scholar-pages', type=str,
    #                     help='If given in %%d format, the number of pages to download from the beginning. '
    #                          'If given in %%d-%%d format, the range of pages (starting from 1) to download (the end is included). '
    #                          'Each page has a maximum of 10 papers (required for --query)')
    # parser.add_argument('--dwn-dir', type=str, help='Directory path in which to save the results')
    # parser.add_argument('--min-year', default=None, type=int, help='Minimal publication year of the paper to download')
    # parser.add_argument('--max-dwn-year', default=None, type=int,
    #                     help='Maximum number of papers to download sorted by year')
    # parser.add_argument('--max-dwn-cites', default=None, type=int,
    #                     help='Maximum number of papers to download sorted by number of citations')
    # parser.add_argument('--journal-filter', default=None, type=str,
    #                     help='CSV file path of the journal filter (More info on github)')
    # parser.add_argument('--restrict', default=None, type=int, choices=[0, 1],
    #                     help='0:Download only Bibtex - 1:Down load only papers PDF')
    # parser.add_argument('--scihub-mirror', default=None, type=str,
    #                     help='Mirror for downloading papers from sci-hub. If not set, it is selected automatically')
    # parser.add_argument('--annas-archive-mirror', default=None, type=str,
    #                     help='Mirror for downloading papers from Annas Archive (SciDB). If not set, https://annas-archive.se is used')
    # parser.add_argument('--scholar-results', default=10, type=int, choices=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    #                     help='Downloads the first x results for each scholar page(default/max=10)')
    # parser.add_argument('--proxy', nargs='+', default=[],
    #                     help='Use proxychains, provide a seperated list of proxies to use.Please specify the argument al the end')
    # parser.add_argument('--single-proxy', type=str, default=None,
    #                     help='Use a single proxy. Recommended if using --proxy gives errors')
    # parser.add_argument('--selenium-chrome-version', type=int, default=None,
    #                     help='First three digits of the chrome version installed on your machine. If provided, selenium will be used for scholar search. It helps avoid bot detection but chrome must be installed.')
    # parser.add_argument('--use-doi-as-filename', action='store_true', default=False,
    #                     help='Use DOIs as output file names')
    # args = parser.parse_args()

    if single_proxy is not None:
        os.environ['http_proxy'] = single_proxy
        os.environ['HTTP_PROXY'] = single_proxy
        os.environ['https_proxy'] = single_proxy
        os.environ['HTTPS_PROXY'] = single_proxy
        print("Using proxy: ", single_proxy)
    else:
        pchain = []
        pchain = proxy
        proxy_func(pchain)

    if query is None and doi_file is None and doi is None and cites is None:
        logging.error(
            "Error: At least one option between 'query', 'doi-file', 'doi' and 'cites' must be used"
        )
        return None

    if (
        (query is not None and doi_file is not None)
        or (query is not None and doi is not None)
        or (doi is not None and doi_file is not None)
    ):
        logging.error(
            "Error: Only one option between 'query', 'doi-file' and 'doi' can be used"
        )
        return None

    if dwn_dir is None:
        logging.error("Error, provide the directory path in which to save the results")
        return None

    if scholar_results != 10 and scholar_pages != 1:
        logging.info("Scholar results best applied along with --scholar-pages=1")

    dwn_dir = dwn_dir.replace('\\', '/')
    if dwn_dir[-1] != '/':
        dwn_dir += "/"
    if not os.path.exists(dwn_dir):
        os.makedirs(dwn_dir, exist_ok=True)

    if max_dwn_year is not None and max_dwn_cites is not None:
        logging.error(
            "Error: Only one option between 'max-dwn-year' and 'max-dwn-cites' can be used "
        )
        return None

    if query is not None or cites is not None:
        if scholar_pages:
            try:
                split = scholar_pages.split('-')
                if len(split) == 1:
                    scholar_pages = range(1, int(split[0]) + 1)
                elif len(split) == 2:
                    start_page, end_page = [int(x) for x in split]
                    scholar_pages = range(start_page, end_page + 1)
                else:
                    raise ValueError
            except Exception:
                logging.error(
                    r"Error: Invalid format for --scholar-pages option. Expected: %d or %d-%d, got: "
                    + args.scholar_pages
                )
                return None
        else:
            logging.error("Error: with --query provide also --scholar-pages")
            return None
    else:
        scholar_pages = 0

    DOIs = None
    if doi_file is not None:
        DOIs = []
        f = doi_file.replace('\\', '/')
        with open(f) as file_in:
            for line in file_in:
                if line[-1] == '\n':
                    DOIs.append(line[:-1])
                else:
                    DOIs.append(line)

    if doi is not None:
        DOIs = [doi]

    max_dwn = None
    max_dwn_type = None
    if max_dwn_year is not None:
        max_dwn = max_dwn_year
        max_dwn_type = 0
    if max_dwn_cites is not None:
        max_dwn = max_dwn_cites
        max_dwn_type = 1

    start_download_pdf(
        query,
        scholar_results,
        scholar_pages,
        dwn_dir,
        proxy,
        min_year,
        max_dwn,
        max_dwn_type,
        journal_filter,
        restrict,
        DOIs,
        scihub_mirror,
        selenium_chrome_version,
        cites,
        use_doi_as_filename,
        annas_archive_mirror,
        skip_words,
    )

    return dwn_dir
