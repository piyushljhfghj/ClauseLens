"""JSON responses that declare their encoding.

Starlette sends `application/json` without a charset. JSON is UTF-8 by definition, but some
clients fall back to ISO-8859-1 when no charset is given (Windows PowerShell 5.1's
Invoke-RestMethod does), which displays every non-ASCII character as mojibake, e.g. a curly
quote as "â". Declaring the charset makes those clients decode the same bytes correctly.
"""

from fastapi.responses import JSONResponse


class UTF8JSONResponse(JSONResponse):
    media_type = "application/json; charset=utf-8"
