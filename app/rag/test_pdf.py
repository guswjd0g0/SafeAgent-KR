import fitz

PDF_PATH = "data/documents/산업재해보상보험법(법률)(제21375호)(20260701).pdf"

doc = fitz.open(PDF_PATH)

print(f"전체 페이지 수: {len(doc)}")

for page_number in range(min(3, len(doc))):
    page = doc[page_number]

    text = page.get_text()

    print("\n" + "=" * 60)
    print(f"PAGE {page_number + 1}")
    print("=" * 60)

    print(text[:2000])