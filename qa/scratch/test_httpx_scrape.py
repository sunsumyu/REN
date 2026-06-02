import urllib.request
import urllib.parse
import ssl
import xml.etree.ElementTree as ET

def test_efetch_xml(pmid: str):
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {
        "db": "pubmed",
        "id": pmid,
        "retmode": "xml"
    }
    req_url = f"{url}?{urllib.parse.urlencode(params)}"
    print(f"Requesting EFetch XML URL:\n{req_url}\n")
    
    req = urllib.request.Request(req_url, headers={'User-Agent': 'Mozilla/5.0'})
    context = ssl._create_unverified_context()
    try:
        with urllib.request.urlopen(req, context=context, timeout=8) as r:
            xml_data = r.read()
            print(f"XML Data length: {len(xml_data)}")
            
            root = ET.fromstring(xml_data)
            
            # Find the PubmedArticle
            for article in root.findall('.//PubmedArticle'):
                # Extract title
                title_el = article.find('.//ArticleTitle')
                title = "".join(title_el.itertext()).strip() if title_el is not None else "Unknown Title"
                
                # Extract publication date
                pubdate_el = article.find('.//JournalIssue/PubDate')
                pubdate = ""
                if pubdate_el is not None:
                    year = pubdate_el.find('Year')
                    month = pubdate_el.find('Month')
                    day = pubdate_el.find('Day')
                    medline = pubdate_el.find('MedlineDate')
                    if year is not None:
                        pubdate = "".join(year.itertext()).strip()
                    elif medline is not None:
                        pubdate = "".join(medline.itertext()).strip()
                
                # Extract authors
                authors = []
                for author in article.findall('.//AuthorList/Author'):
                    last_name = author.find('LastName')
                    fore_name = author.find('ForeName')
                    initials = author.find('Initials')
                    if last_name is not None:
                        ln = "".join(last_name.itertext()).strip()
                        fn = "".join(fore_name.itertext()).strip() if fore_name is not None else ""
                        authors.append(f"{ln} {fn}".strip())
                author_str = ", ".join(authors) if authors else "Unknown Authors"
                
                # Extract journal/source
                journal_el = article.find('.//Journal/Title')
                journal = "".join(journal_el.itertext()).strip() if journal_el is not None else "Unknown Journal"
                
                # Extract abstract
                abstract_texts = []
                for abs_text in article.findall('.//Abstract/AbstractText'):
                    label = abs_text.attrib.get('Label', '')
                    text = "".join(abs_text.itertext()).strip()
                    if label:
                        abstract_texts.append(f"{label}: {text}")
                    else:
                        abstract_texts.append(text)
                abstract = "\n".join(abstract_texts) if abstract_texts else "No abstract available."
                
                print(f"Title: {title}")
                print(f"PubDate: {pubdate}")
                print(f"Authors: {author_str}")
                print(f"Journal: {journal}")
                print(f"Abstract Snippet: {abstract[:200]}...")
                
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_efetch_xml("42014325")
