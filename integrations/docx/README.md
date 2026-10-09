# foveate-docx

Word (.docx) loader for [Foveate](https://github.com/sachncs/foveate). Foveate's core is pure Python with no
dependencies; formats that need a third-party library live in packages like this one.

Install it and Foveate finds it automatically (entry point `foveate.loaders`):

```python
from foveate import Document

Document.load("file.docx")
```
