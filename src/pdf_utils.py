from PyPDF2 import PdfReader

def extract_text_from_pdf(path: str) -> str:
    """
    Extrait le texte de toutes les pages d'un PDF et le renvoie
    comme une seule grande chaîne.
    """
    reader = PdfReader(path)
    texts = []
    for page in reader.pages:
        t = page.extract_text()
        if t:
            texts.append(t.strip())
    return "\n\n".join(texts)
