class IngestionError(Exception):
    """Base class for all ingestion problems."""


class UnsupportedFileTypeError(IngestionError):
    pass


class FileTooLargeError(IngestionError):
    pass


class EmptyDocumentError(IngestionError):
    """No extractable text (e.g. a scanned PDF or an empty file)."""


class CorruptDocumentError(IngestionError):
    """The file could not be opened or parsed."""