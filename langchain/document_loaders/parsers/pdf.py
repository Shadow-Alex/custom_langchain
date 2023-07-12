"""Module contains common parsers for PDFs."""
import time
from typing import Any, Iterator, Mapping, Optional, Union
import cv2
from langchain.document_loaders.base import BaseBlobParser
from langchain.document_loaders.blob_loaders import Blob
from langchain.schema import Document


class PyPDFParser(BaseBlobParser):
    """Loads a PDF with pypdf and chunks at character level."""

    def __init__(self, password: Optional[Union[str, bytes]] = None):
        self.password = password

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        """Lazily parse the blob."""
        import pypdf

        with blob.as_bytes_io() as pdf_file_obj:
            pdf_reader = pypdf.PdfReader(pdf_file_obj, password=self.password)
            yield from [
                Document(
                    page_content=page.extract_text(),
                    metadata={"source": blob.source, "page": page_number},
                )
                for page_number, page in enumerate(pdf_reader.pages)
            ]


class PDFMinerParser(BaseBlobParser):
    """Parse PDFs with PDFMiner."""

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        """Lazily parse the blob."""
        from pdfminer.high_level import extract_text

        with blob.as_bytes_io() as pdf_file_obj:
            text = extract_text(pdf_file_obj)
            metadata = {"source": blob.source}
            yield Document(page_content=text, metadata=metadata)


class PyMuPDFParser(BaseBlobParser):
    """Parse PDFs with PyMuPDF."""

    def __init__(self, text_kwargs: Optional[Mapping[str, Any]] = None) -> None:
        """Initialize the parser.

        Args:
            text_kwargs: Keyword arguments to pass to ``fitz.Page.get_text()``.
        """
        self.text_kwargs = text_kwargs or {}

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        """Lazily parse the blob."""
        import fitz

        with blob.as_bytes_io() as file_path:
            doc = fitz.open(file_path)  # open document

            yield from [
                Document(
                    page_content=page.get_text(**self.text_kwargs),
                    metadata=dict(
                        {
                            "source": blob.source,
                            "file_path": blob.source,
                            "page": page.number,
                            "total_pages": len(doc),
                        },
                        **{
                            k: doc.metadata[k]
                            for k in doc.metadata
                            if type(doc.metadata[k]) in [str, int]
                        },
                    ),
                )
                for page in doc
            ]


class PyPDFium2Parser(BaseBlobParser):
    """Parse PDFs with PyPDFium2."""

    def __init__(self) -> None:
        """Initialize the parser."""
        try:
            import pypdfium2  # noqa:F401
        except ImportError:
            raise ValueError(
                "pypdfium2 package not found, please install it with"
                " `pip install pypdfium2`"
            )

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        """Lazily parse the blob."""
        import pypdfium2

        # pypdfium2 is really finicky with respect to closing things,
        # if done incorrectly creates seg faults.
        with blob.as_bytes_io() as file_path:
            pdf_reader = pypdfium2.PdfDocument(file_path, autoclose=True)
            try:
                for page_number, page in enumerate(pdf_reader):
                    text_page = page.get_textpage()
                    content = text_page.get_text_range()
                    text_page.close()
                    page.close()
                    metadata = {"source": blob.source, "page": page_number}
                    yield Document(page_content=content, metadata=metadata)
            finally:
                pdf_reader.close()


class PDFPlumberParser(BaseBlobParser):
    """Parse PDFs with PDFPlumber."""

    def __init__(self, text_kwargs: Optional[Mapping[str, Any]] = None) -> None:
        """Initialize the parser.

        Args:
            text_kwargs: Keyword arguments to pass to ``pdfplumber.Page.extract_text()``
        """
        self.text_kwargs = text_kwargs or {}

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        """Lazily parse the blob."""
        import pdfplumber

        with blob.as_bytes_io() as file_path:
            doc = pdfplumber.open(file_path)  # open document

            yield from [
                Document(
                    page_content=page.extract_text(**self.text_kwargs),
                    metadata=dict(
                        {
                            "source": blob.source,
                            "file_path": blob.source,
                            "page": page.page_number,
                            "total_pages": len(doc.pages),
                        },
                        **{
                            k: doc.metadata[k]
                            for k in doc.metadata
                            if type(doc.metadata[k]) in [str, int]
                        },
                    ),
                )
                for page in doc.pages
            ]


class Pix2TextParser(BaseBlobParser):
    def __init__(self) -> None:
        """Initialize the parser."""
        try:
            from pix2text import Pix2Text  # noqa:F401
        except ImportError:
            raise ValueError(
                "Pix2Text package not found, please install it with"
                " `pip install pix2text`"
            )

        self.p2t = Pix2Text(
            analyzer_config=dict(model_name='mfd'),
            general_config=dict(rec_model_backend='pytorch', det_model_backend='pytorch'),
            device='cuda')

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        """Lazily parse the blob."""
        import fitz, os

        file_path = blob.path
        doc = fitz.open(file_path)
        pdf_name_ext = os.path.basename(file_path)
        pdf_name = os.path.splitext(pdf_name_ext)[0]

        for idx, page in enumerate(doc):
            pix = page.get_pixmap(dpi=300)  # render page to an image
            pix.save(pdf_name + '_page_' + str(idx) + '.png')

            outs = self.p2t(pdf_name + '_page_' + str(idx) + '.png', resized_shape=700)

            header_text = ''
            footer_text = ''
            body_text = ''

            for out in outs:
                if out['type'] == 'Footer':
                    footer_text += out['text']
                elif out['type'] == 'Header':
                    header_text += out['text']
                elif out['type'] == 'Reference':
                    continue
                else:
                    body_text += out['text']

            only_text = header_text + '\n\n' + body_text + '\n\n' + footer_text + '\n\n'

            # 在这里处理单列版面 vs 双列版面。
            # 二者唯一的区别是，当我们从左向右，从上到下地遍历文档时。
            # 双列版面中，那些在版面中线右边的 box 应当暂时保留在 cache 中，直到遇到一个横跨中线的 box，才清空 cache

            os.remove(pdf_name + '_page_' + str(idx) + '.png')

            yield Document(
                page_content=only_text,
                metadata={'source': blob.source, "page": idx + 1}
            )