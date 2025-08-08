# 🕷️ Web Crawler + AI Categorizer & Summarizer

A Python application that **crawls websites**, categorizes and summarizes their pages using **Mistral AI**, and stores the results in **Supabase** for easy querying and analysis.

This tool is ideal for **content analysis**, **SEO research**, or **building structured datasets** from unstructured web pages.

---

## 🚀 Features

- **Web scraping with Crawl4AI**  
  Automatically crawls and extracts content, links, and metadata from target websites.

- **AI-powered categorization**  
  Classifies pages (e.g., blog posts, product pages, documentation) using **Mistral**.

- **Page summarization**  
  Generates concise, human-readable summaries of each page.

- **Metadata extraction**  
  Captures titles, descriptions, authors, publish dates, and canonical URLs.

- **Supabase integration**  
  Saves all processed data (URL, category, summary, metadata) into a Supabase table.

---

## 🛠️ Tech Stack

- **[Python 3.10+](https://www.python.org/)**
- **[Crawl4AI](https://github.com/crawl4ai/crawl4ai)** — Web crawling & content extraction
- **[Mistral API](https://docs.mistral.ai/)** — Natural language processing for categorization & summarization
- **[Supabase](https://supabase.com/)** — Cloud database for storing results

---

## 📦 Installation

bash
git clone https://github.com/yourusername/web-crawler-mistral-supabase.git
cd web-crawler-mistral-supabase

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install -r requirements.txt


Configuration
Create a .env file in the project root:

# Mistral API Key
MISTRAL_API_KEY=your_mistral_api_key_here

# Supabase credentials
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your_supabase_service_key

# Crawl4AI settings
CRAWL4AI_USER_AGENT=Mozilla/5.0 (compatible; MyCrawler/1.0)

# Target site
TARGET_URL=https://example.com

#The script will:

Crawl the target website using Crawl4AI.

For each page:

Extract URL, title, and metadata.

Use Mistral to:

Categorize the page (e.g., blog, docs, product).

Summarize the content.

Save results into Supabase.

