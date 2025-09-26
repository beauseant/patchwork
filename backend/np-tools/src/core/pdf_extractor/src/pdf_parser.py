"""
Class for parsing PDF files. It extracts the text, tables, and images from the PDF and generates a JSON file with the extracted content. It also generates a description for the images and tables in the PDF, if specified. The extracted JSON structure is as follows:
{
    "metadata": {...},
    "header": "...",
    "footer": "...",
    "pages": [
        {
            "page_number": 0,
            "content": [
                {
                    "element_id": 0,
                    "element_type": "text",
                    "element_content": "...",
                    "element_description": null,
                    "element_path": null
                },
                {
                    "element_id": 1,
                    "element_type": "table",
                    "element_content": "...",
                    "element_description": null,
                    "element_path": "path/to/table"
                },
                {
                    "element_id": 2,
                    "element_type": "image",
                    "element_content": null,
                    "element_description": "...",
                    "element_path": "path/to/image"
                }
            ]
        }
    ]
}
Element 0 from each page is the text extracted from the page. Element 1 and onward are the tables and images extracted from the page, if any, maintaining the order they appear in the page. 

The code is based on the following sources:
* https://github.com/g-stavrakis/PDF_Text_Extraction/tree/main for the text / tables / images extraction
* https://medium.com/@hussainshahbazkhawaja/paper-implementation-header-and-footer-extraction-by-page-association-3a499b2552ae for the header / footer extraction mechanism

Author: Lorena Calvo-Bartolomé
Date: 04/02/2024
"""

import json
import logging
import os
import pathlib
import re
import time
from typing import Tuple

import fitz
import pandas as pd
import pdfplumber
import PyPDF2
from dotenv import load_dotenv
from fuzzywuzzy import fuzz
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from openai import OpenAI
from pdf2image import convert_from_path
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTFigure, LTPage, LTTextContainer
from src.image_descriptor import ImageDescriptor
from src.multi_column import column_boxes
from src.utils import compare
from tqdm import tqdm


class PDFParser(object):
    def __init__(
        self,
        extract_header_footer: bool = True,
        generate_img_desc: bool = False,
        generate_table_desc: bool = False,
        header_weights: list = [1.0, 0.75, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        footer_weights: list = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.75, 1.0],
        win: int = 8,
        footer_marging: int = 50,
        header_marging: int = 50,
        remove_next_line: bool = True
    ) -> None:
        """
        Initialize the PDFParser object.

        Parameters
        ----------
        extract_header_footer : bool, optional
            Whether to extract the header and footer from the PDF, by default True
        generate_img_desc : bool, optional
            Whether to generate a description for the images in the PDF, by default False
        generate_table_desc : bool, optional
            Whether to generate a description for the tables in the PDF, by default False
        header_weights : list, optional
            The weights for the header, by default [1.0, 0.75, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        footer_weights : list, optional
            The weights for the footer, by default [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.75, 1.0]
        win : int, optional
            Parameter to control the number of neighboring pages in the header / footer extraction mechanism by default 8
        footer_marging : int, optional
            The margin to consider for the footer extraction when PyMuPDF's  multi_column.py utility is used, by default 50
        header_marging : int, optional
            The margin to consider for the header extraction when PyMuPDF's  multi_column.py utility is used, by default 50
        remove_next_line : bool, optional
            Whether to remove the next line character from the text, by default True
        """

        # Create a logger
        logging.basicConfig(level='INFO')
        self._logger = logging.getLogger(__name__)

        # Get OpenAI API key from .env file
        path_env = pathlib.Path(os.getcwd()).parent / '.env'
        load_dotenv(path_env)
        self._api_key = os.getenv("OPENAI_API_KEY")

        # Initialize image descriptor if generate_img_desc is True
        if generate_img_desc:
            self._image_descriptor = ImageDescriptor()
            self._logger.info(
                f"-- -- Image descriptor initialized with OpenAI API key."
            )

        # Initialize the variables
        self._extract_header_footer = extract_header_footer
        self._generate_img_desc = generate_img_desc
        self._generate_table_desc = generate_table_desc
        self._header_weights = header_weights
        self._footer_weights = footer_weights
        self._win = win
        self._footer_marging = footer_marging
        self._header_marging = header_marging
        self._table_counter = 0
        self._img_counter = 0
        self._remove_next_line = remove_next_line

    # ======================================================
    # HEADER / FOOTER EXTRACTION
    # ======================================================
    def _extract_text_from_page(
        self,
        page
    ) -> list:
        """Extract the text from the page given by page for the header / footer extraction.

        Parameters
        ----------
        page : fitz.fitz.Page
            The page to extract the text from

        Returns
        -------
        list
            The text extracted from the page
        """

        text = page.get_text(sort=True)
        text = text.split('\n')
        text = [t.strip() for t in text if t.strip()]

        return text

    def _extract_header(
        self,
        header_candidates: list
    ):
        """Extract the header from the header_candidates.

        Parameters
        ----------
        header_candidates : list
            The list of header candidates

        Returns
        -------
        list
            The extracted header
        """

        self._logger.info(
            f"-- Extracting header from header candidates...")

        all_detected = []
        for i, candidate in enumerate(header_candidates):
            temp = header_candidates[max(
                i-self._win, 1): min(i+self._win, len(header_candidates))]
            if temp == []:  # Consider pdf with only one page
                temp = header_candidates
            maxlen = len(max(temp, key=len))
            for sublist in temp:
                sublist[:] = sublist + [''] * (maxlen - len(sublist))
            detected = []
            for j, cn in enumerate(candidate):
                score = 0
                try:
                    cmp = list(list(zip(*temp))[j])
                    for cm in cmp:
                        score += compare(cn, cm) * self._header_weights[j]
                    score = score/len(cmp)
                except:
                    score = self._header_weights[j]
                if score > 0.5:
                    detected.append(cn)
            del temp

            all_detected.extend(detected)

        detected = list(set(all_detected))

        return detected

    def _extract_footer(
        self,
        footer_candidates: list
    ) -> list:
        """
        Extract the footer from the footer_candidates.

        Parameters
        ----------
        footer_candidates : list
            The list of footer candidates

        Returns
        -------
        list
            The extracted footer
        """

        self._logger.info(
            f"-- Extracting footer from footer candidates...")

        all_detected = []
        for i, candidate in enumerate(footer_candidates):
            temp = footer_candidates[max(
                i-self._win, 1): min(i+self._win, len(footer_candidates))]
            if temp == []:  # Consider pdf with only one page
                temp = footer_candidates
            maxlen = len(max(temp, key=len))
            for sublist in temp:
                sublist[:] = [''] * (maxlen - len(sublist)) + sublist
            detected = []
            for j, cn in enumerate(candidate):
                score = 0
                try:
                    cmp = list(list(zip(*temp))[j])
                    for cm in cmp:
                        score += compare(cn, cm)
                    score = score/len(cmp)
                except:
                    score = self._footer_weights[j]
                if score > 0.5:
                    detected.append(cn)
            del temp

            all_detected.extend(detected)

        detected = list(set(all_detected))

        return detected

    # ======================================================
    # TEXT / TABLE / IMAGE EXTRACTION
    # ======================================================
    def _parse_text(
        self,
        text: str,
    ) -> str:
        """
        Parse the text extracted from the PDF in order to make it more readable.

        Parameters
        ----------
        text : str
            The text to parse

        Returns
        -------
        str
            The parsed text
        """

        text_formatted = re.sub(' +', ' ', text).strip()
        
        # Remove \n
        if self._remove_next_line:
            text_formatted = text_formatted.replace('\n', ' ')

        # Remove header and footer from the text
        for el in self._header + self._footer:
            text_formatted = text_formatted.replace(
                el, "").replace('-', '–').replace(el, "")

        if text_formatted == ".":
            text_formatted = ""

        return text_formatted

    def _extract_table(
        self,
        pdf_path: str,
        page_num: int,
        table_num: int
    ) -> list:
        """Extract the table given by table_num from the page given by page_num from the pdf given by pdf_path.

        Parameters
        ----------
        pdf_path : str
            The path to the pdf file
        page_num : int
            The page number of the page to extract the table from
        table_num : int
            The table number of the table to extract from the page

        Returns
        -------
        list
            The table extracted from the pdf
        """

        # Open the pdf file
        pdf = pdfplumber.open(pdf_path)
        # Find the examined page
        table_page = pdf.pages[page_num]
        # Extract the appropriate table
        table = table_page.extract_tables()[table_num]

        return table

    def _get_label_table(
        self,
        table: pd.DataFrame
    ) -> str:
        """Get the label for the table given by table.

        Parameters
        ----------
        table : pd.DataFrame
            The table to describe

        Returns
        -------
        str
            The description of the table
        """

        parameters = {
            "model": "gpt-4",
            "messages": [
                {"role": "system", "content": f"You are a helpful assistant trained on the task of giving a descriptive label to a table given in DataFrame format. If characterized by several words, unify it with underscores."
                 },
            ],
            "temperature": 0.2,
            "max_tokens": 1000,
            "frequency_penalty": 0.0
        }

        parser = OpenAI(api_key=self._api_key)

        gpt_prompt = f"Give me a label to save this table.: {table}"

        message = [{"role": "user", "content": gpt_prompt}]
        parameters["messages"] = [parameters["messages"][0], *message]

        response = parser.chat.completions.create(
            **parameters
        )
        label = response.choices[0].message.content

        return label

    def _table_converter(
        self,
        table: list,
        pageNr: int,
        path_save: pathlib.Path
    ) -> str:
        """Convert the table given by table into a string.

        Parameters
        ----------
        table : list
            The table to convert
        pageNr : int
            The page number of the page the table is on

        Returns
        -------
        str
            The table converted into a string
        """

        table_output_save = None
        description = None

        # Convert to string
        table_string = ""
        # Iterate through each row of the table
        cleaned_table = []
        for row_num in range(len(table)):
            row = table[row_num]
            # Remove the line breaker from the wrapted texts
            cleaned_row = [
                (item.replace("\n", " ")).strip()
                if item is not None
                else ""
                if item is None
                else item
                for item in row
            ]
            # Convert the table into a string
            table_string += " ".join(cleaned_row)
            cleaned_table.append(cleaned_row)

        # Check if the table is only composed by the header or the footer
        aux = table_string
        for el in self._header + self._footer:
            aux = aux.replace(el, "").strip()
            if not aux:
                break

        if aux == "" or aux is None:
            return None, None, None
        else:
            for el in self._header + self._footer:
                table_string = table_string.replace(el, "")

            table_string = self._parse_text(table_string)

            for row in cleaned_table[:]:
                remove_row = False
                for el_hf in self._header + self._footer:
                    for i, el in enumerate(row):
                        if el and fuzz.ratio(el_hf, el) > 50:
                            row[i] = el.replace(el, "")
                            remove_row = True
                all_null_none = all(el is None or el == "" for el in row)
                if remove_row and all_null_none:
                    cleaned_table.remove(row)

            if len(cleaned_table) > 1:
                # Convert to dataframe an save as csv/excel
                df_table = pd.DataFrame(cleaned_table)

                # Generate a label to save the table if generate_table_desc is True; otherwise, use a counter
                if self._generate_table_desc:
                    label = self._get_label_table(df_table)
                else:
                    label = str(self._table_counter)
                    self._table_counter += 1
                description = None
                table_output_save = f"{path_save.as_posix()}/tables/page_{pageNr}_{label}.xlsx"
                df_table.to_excel(table_output_save)
            else:
                return None, None, None

        return table_string, table_output_save, description

    def _is_text_element_inside_any_table(
        self,
        element: LTTextContainer,
        page: LTPage,
        tables: list
    ) -> bool:
        """
        Check if the element is in any tables present in the page.

        Parameters
        ----------
        element : LTTextContainer
            The element to check
        page : LTPage
            The page the element is on
        tables : list
            The list of tables on the page

        Returns
        -------
        bool
            True if the element is inside any table, False otherwise
        """

        x0, y0up, x1, y1up = element.bbox
        # Change the cordinates because the pdfminer counts from the botton to top of the page
        y0 = page.bbox[3] - y1up
        y1 = page.bbox[3] - y0up
        for table in tables:
            tx0, ty0, tx1, ty1 = table.bbox
            if tx0 <= x0 <= x1 <= tx1 and ty0 <= y0 <= y1 <= ty1:
                return True
        return False

    def _find_table_for_element(
        self,
        element: LTTextContainer,
        page: LTPage,
        tables: list
    ) -> int:
        """Find the table for a given element. If the element is not inside any table, return None.

        Parameters
        ----------
        element : LTTextContainer
            The element to find the table for
        page : LTPage
            The page the element is on
        tables : list
            The list of tables on the page
        """

        x0, y0up, x1, y1up = element.bbox
        # Change the cordinates because the pdfminer counts from the botton to top of the page
        y0 = page.bbox[3] - y1up
        y1 = page.bbox[3] - y0up
        for i, table in enumerate(tables):
            tx0, ty0, tx1, ty1 = table.bbox
            if tx0 <= x0 <= x1 <= tx1 and ty0 <= y0 <= y1 <= ty1:
                return i  # Return the index of the table
        return None

    def _extract_image(
        self,
        element: LTFigure,
        pageObj: LTPage,
        pageNr: int,
        path_save: pathlib.Path,
        table_info: str = None
    ) -> Tuple[str, str, str]:
        """Extract the image given by element from the page given by pageObj from the pdf given by pdf_path. Generate a textual description of the image and save it with a label.

        Parameters
        ----------
        element : LTImage
            The image to extract from the page
        pageObj : LTPage
            The page to extract the image from
        pageNr : int
            The page number of the page to extract the image from
        table_info : str, optional
            The information of the table the image is in, by default None

        Returns
        -------
        Tuple[str, str, str]
            The path to the image, the textual description of the image, and the label of the image
        """

        # Get the coordinates to crop the image from PDF
        [image_left, image_top, image_right, image_bottom] = [
            element.x0, element.y0, element.x1, element.y1]
        # Crop the page using coordinates (left, bottom, right, top)
        pageObj.mediabox.lower_left = (image_left, image_bottom)
        pageObj.mediabox.upper_right = (image_right, image_top)
        # Save the cropped page to a new PDF
        cropped_pdf_writer = PyPDF2.PdfWriter()
        cropped_pdf_writer.add_page(pageObj)

        # Save the cropped PDF to a new file
        pdf_image_save = path_save / "cropped_image.pdf"
        with open(pdf_image_save, "wb") as cropped_pdf_file:
            cropped_pdf_writer.write(cropped_pdf_file)

        # Temporarily convert the PDF and save as image
        images = convert_from_path(pdf_image_save)
        output_file = path_save / "images" / "cropped_image.png"
        images[0].save(output_file, 'PNG')

        # Get the a textual label to save the image
        if self._generate_img_desc:
            self._logger.info(
                f"-- Waiting 3 minutes to avoid OpenAI API rate limit...")
            time.sleep(60*3)
            # Get the a textual label to save the image
            label = self._image_descriptor.get_label_image(
                pathlib.Path(output_file))

            if not "logo" in label:
                # If the label is not "logo" (the image does not represent a logo according to gpt-4), get the textual description of the image
                self._logger.info(
                    f"-- Waiting 3 minutes to avoid OpenAI API rate limit...")
                time.sleep(60*3)
                description = self._image_descriptor.describe_image(
                    pathlib.Path(output_file))
            else:
                description = "logo"
        else:
            description = None
            label = str(self._img_counter)
            self._img_counter += 1

        # Save the image with name "page_{pageNr}_{label}.png"
        if table_info:
            # If the image is inside a table, add the table information to the label
            label = f"{table_info}_{label}"
            new_output_file = f"{path_save.as_posix()}/images/{label}.png"
        else:
            new_output_file = f"{path_save.as_posix()}/images/page_{pageNr}_{label}.png"

        images[0].save(new_output_file, 'PNG')

        # Remove the cropped PDF and temporal image
        os.remove(pdf_image_save)
        os.remove(output_file)

        return new_output_file, description, label

    def _get_metadata(
        self,
        pdf_path: pathlib.Path
    ) -> dict:
        """Get the metadata of the PDF given by pdf_path.

        Parameters
        ----------
        pdf_path : pathlib.Path
            The path to the PDF file

        Returns
        -------
        dict
            The metadata of the PDF
        """

        loader = PyMuPDFLoader(pdf_path.as_posix())

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000,
            chunk_overlap=200,
        )

        langchain_docs2 = loader.load_and_split(text_splitter)
        return langchain_docs2[0].metadata

    def _generate_json(
        self,
        content_per_page: list
    ) -> dict:
        """Generate a JSON file with the extracted content from the PDF.

        Parameters
        ----------
        content_per_page : list
            The content extracted from each page of the PDF

        Returns
        -------
        dict
            The JSON file with the extracted content
        """
        document_json = {
            "metadata": self._metadata,
            "header": " ".join(self._header),
            "footer": " ".join(self._footer),
            "pages": content_per_page
        }
        return document_json

    def parse(
        self,
        pdf_path: pathlib.Path,
        path_save: pathlib.Path
    ) -> dict:
        """
        Parse the PDF given by pdf_path and save the extracted content in a JSON file.

        Parameters
        ----------
        pdf_path : pathlib.Path
            The path to the PDF file
        path_save : pathlib.Path
            The path to save the extracted content

        Returns
        -------
        dict
            The JSON file with the extracted content 
        """

        ########################################################################
        # 1. Extract metadata
        # ----------------------------------------------------------------------
        # This refers to the information about the PDF file, such as the title, author, subject, and keywords, that is stored in the document properties.
        ########################################################################
        self._metadata = self._get_metadata(pdf_path)

        ########################################################################
        # 2. Extract the header and footer from the PDF
        ########################################################################
        # Open the PDF with PyMuPDF and extract the text from the pages
        pages = fitz.open(pdf_path)
        pages_ = [self._extract_text_from_page(page) for page in pages]

        if self._extract_header_footer:
        # Get the header and footer candidates
            header_candidates = [page[:self._win] for page in pages_]
            footer_candidates = [page[-self._win:] for page in pages_]

            # Extract the header and footer from the header and footer candidates
            self._header = self._extract_header(header_candidates)
            self._footer = self._extract_footer(footer_candidates)
        else:
            self._header = []
            self._footer = []

        ########################################################################
        # 3. Extract the content from the PDF
        ########################################################################
        # Create a PDF file object
        pdfFileObj = open(pdf_path, 'rb')

        # Create a PDF reader object
        pdfReaded = PyPDF2.PdfReader(pdfFileObj)

        # Create the dictionary where the content extracted from each PDF page will be stored
        content_per_page = []

        # Initialize the number of the examined tables
        table_in_page = -1

        # Iterate through each page. We keep objects from fitz and pypdf2 to extract so both content (including multicolumns) and tables/images can be extracted
        for pagenum, (page_fitz, page_pypdf2) in enumerate(zip(pages, extract_pages(pdf_path))):
            
            # Reset the table and image counters at the beginning of each page
            self._table_counter = 0
            self._img_counter = 0

            # Reset variable to store information about a page's elements
            element_id = 0

            # Table extraction
            self._logger.info(
                f"-- Table extraction from page {pagenum} starts...")

            # Initialize variables
            pageObj = pdfReaded.pages[pagenum]
            text_from_tables = []
            path_tables = []
            description_tables = []
            this_page_content = []

            # Open the PDF file with pdfplumber to extract tables
            pdf = pdfplumber.open(pdf_path)

            # Find the pdfplumber page
            page_tables = pdf.pages[pagenum]

            # Find the number of tables in the page
            tables = page_tables.find_tables()
            if len(tables) != 0:
                table_in_page = 0
                self._logger.info(f"-- Extracting tables from page {pagenum}...")
            else:
                self._logger.info(f"-- No tables found in page {pagenum}.")

            # Extracting the tables of the page
            for i_table in tqdm(range(len(tables))):

                # Extract the information of the table
                table = self._extract_table(pdf_path, pagenum, i_table)

                # Convert the table information in structured string format, save the table as image and get the description
                table_string, table_output_save, description = \
                    self._table_converter(table, pagenum, path_save)

                text_from_tables.append(table_string)
                path_tables.append(table_output_save)
                description_tables.append(description)
                
                if i_table == len(tables) - 1:
                    self._logger.info(
                        f"-- Table extraction from page {pagenum} finished...")

            # Find all the elements and sort them as they appear in the page
            page_elements = [(element.y1, element) for element in page_pypdf2._objs]
            page_elements.sort(key=lambda a: a[0], reverse=True)

            # Extract text content from the page with the multi_column utility
            # We remove the tables from the page content
            if table_in_page != -1:
                [page_fitz.add_redact_annot(table.bbox) for table in page_fitz.find_tables()]
            page_fitz.apply_redactions()

            # Extract columns (if any)
            bboxes = column_boxes(page_fitz, footer_margin=self._footer_marging, no_image_text=True)

            # If columns, concatenate the text from the columns
            if len(bboxes) > 1:
                self._logger.info(f"-- Columns found in page {pagenum}...")
                extracted_text_fitz = ''.join([page_fitz.get_text(clip=rect, sort=True) for rect in bboxes])
            else:
                # If no columns, extract the text from the page
                self._logger.info(f"-- No columns found in page {pagenum}...")
                extracted_text_fitz = page_fitz.get_text()

            extracted_text_fitz = self._parse_text(extracted_text_fitz)
            
            # Append the text of each line to the page text as one 'text' element
            this_page_content.append(
                {
                    "element_id": element_id,
                    "element_type": "text",
                    "element_content": extracted_text_fitz,
                    "element_description": None,
                    "element_path": None
                }
            )

            element_id += 1

            self._logger.info(
                f"-- Element extraction from page {pagenum} starts...")

            # Find the elements that compose a page
            for _, component in tqdm(enumerate(page_elements)):

                # Get the element
                element = component[1]

                # Check the elements for tables
                if table_in_page == -1:
                    pass
                else:
                    if self._is_text_element_inside_any_table(element, page_pypdf2, tables):

                        table_idx = self._find_table_for_element(
                            element, page_pypdf2, tables)

                        if table_idx is not None and table_idx == table_in_page:
                            # If a table is found and it is located in the same page as we are currently extracting the content, we add the table to the content
                            # We do not append them if the table is the header or the footer
                            if text_from_tables[table_idx] is not None:
                                this_page_content.append(
                                    {
                                        "element_id": element_id,
                                        "element_type": "table",
                                        "element_content": text_from_tables[table_idx],
                                        "element_description": description_tables[table_idx],
                                        "element_path": path_tables[table_idx]
                                    }
                                )
                                element_id += 1
                            table_in_page += 1

                        # Pass this iteration because the content of this element was extracted from the tables
                        if isinstance(element, LTFigure):

                            table_label = None
                            if path_tables[table_idx]:
                                table_label = f"in_table_{pathlib.Path(path_tables[table_idx]).stem}"

                            # Extract the image and its description
                            image_path, description, label = \
                                self._extract_image(
                                    element, pageObj, pagenum, path_save, table_label)

                            if "logo" in label:
                                os.remove(image_path)

                            else:
                                # Append the image and its description to the content
                                this_page_content.append(
                                    {
                                        "element_id": element_id,
                                        "element_type": "image",
                                        "element_content": None,
                                        "element_description": description,
                                        "element_path": image_path
                                    }
                                )
                                element_id += 1

                        continue

                if not self._is_text_element_inside_any_table(element, page_pypdf2, tables):

                    if isinstance(element, LTFigure):

                        # Extract the image and its description
                        image_path, description, _ = self._extract_image(
                            element, pageObj, pagenum, path_save)

                        # Append the image and its description to the content
                        this_page_content.append(
                            {
                                "element_id": element_id,
                                "element_type": "image",
                                "element_content": None,
                                "element_description": description,
                                "element_path": image_path
                            }
                        )
                        element_id += 1

            content_per_page.append(
                {
                    "page_number": pagenum,
                    "content": this_page_content
                }
            )

        document_json = self._generate_json(content_per_page)
        path_save_json = path_save / f"{pdf_path.stem}.json"
        with open(path_save_json.as_posix(), "w", encoding="utf-8") as json_file:
            json.dump(document_json, json_file, indent=2, ensure_ascii=False)

        return document_json
