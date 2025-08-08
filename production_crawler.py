import asyncio
import os
import json
from collections import deque
from urllib.parse import urljoin, urlparse
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import httpx

# Import required modules
from crawl4ai import (
    AsyncWebCrawler,
)  # The LLM class is not part of the public API, so we will not import it.
from supabase import create_client, Client

# --- Load Environment Variables ---
load_dotenv()

# --- Configuration ---
START_URL = os.getenv("START_URL")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
MAX_PAGES_TO_CRAWL = 100

# --- Pydantic Models for Structured LLM Output ---


class PageClassification(BaseModel):
    """Defines the structure for page classification output."""

    page_type: str = Field(
        description="The primary purpose of the page. Must be one of: Blog, Info, Contact, Other.",
    )


class BlogDetails(BaseModel):
    """Defines the structure for extracted blog details."""

    title: str = Field(
        description="The main article title, extracted even from non-standard tags like <div> or <span>."
    )
    summary: str = Field(description="A concise, one-paragraph summary of the article.")
    content: str = Field(
        description="The full, complete, and unabridged text of the article with all HTML tags removed."
    )


# --- Helper Functions ---


async def check_mistral_api_key(api_key: str, model: str):
    """Makes a direct, simple call to the Mistral API to validate the key."""
    if not api_key:
        return False, "MISTRAL_API_KEY environment variable is not set."

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    # Use a minimal payload with the configured model to ensure it's supported
    data = {
        "model": model,
        "messages": [{"role": "user", "content": "test"}],
        "max_tokens": 2,
    }
    url = "https://api.mistral.ai/v1/chat/completions"

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers, json=data, timeout=15)

        if response.status_code == 200:
            return True, "API key and model are valid."
        else:
            try:
                error_details = response.json()
                message = error_details.get("message", response.text)
            except json.JSONDecodeError:
                message = response.text
            return (
                False,
                f"API returned status {response.status_code}. Message: {message}",
            )
    except httpx.RequestError as e:
        return False, f"A network error occurred: {e}"
    except Exception as e:
        return False, f"An unexpected error occurred: {e}"


async def extract_with_llm(
    api_key: str,
    model: str,
    prompt_template: str,
    context: str,
    pydantic_model: type[BaseModel],
):
    """
    Makes a direct call to the Mistral API for structured data extraction.
    Returns an instance of the Pydantic model on success, or None on failure.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    # Limit context size to avoid exceeding model limits
    full_prompt = (
        f"{prompt_template}\n\nHere is the HTML content to analyze:\n{context[:12000]}"
    )

    data = {
        "model": model,
        "messages": [{"role": "user", "content": full_prompt}],
        "response_format": {"type": "json_object"},  # Use Mistral's native JSON mode
    }
    url = "https://api.mistral.ai/v1/chat/completions"

    try:
        async with httpx.AsyncClient() as client:
            # Use a longer timeout for potentially complex extractions
            response = await client.post(url, headers=headers, json=data, timeout=60)

        if response.status_code == 200:
            try:
                response_data = response.json()
                json_content_str = response_data["choices"][0]["message"]["content"]
                parsed_json = json.loads(json_content_str)
                return pydantic_model.model_validate(parsed_json)
            except (json.JSONDecodeError, KeyError, IndexError) as e:
                print(
                    f"  [!] LLM extraction failed to parse response. Error: {e}. Response: {response.text[:200]}"
                )
                return None
            except Exception as e:  # Catches Pydantic validation errors
                print(
                    f"  [!] LLM extraction failed Pydantic validation. Error: {e}. Response: {response.text[:200]}"
                )
                return None
        else:
            # Handle API errors with specific messages
            try:
                error_details = response.json()
                message = error_details.get("message", response.text)
            except json.JSONDecodeError:
                message = response.text
            print(
                f"  [!] LLM extraction failed. API status {response.status_code}. Message: {message}"
            )
            return None
    except httpx.RequestError as e:
        print(f"  [!] LLM extraction failed due to a network error: {e}")
        return None
    except Exception as e:
        print(f"  [!] An unexpected error occurred during LLM extraction: {e}")
        return None


async def initialize_clients():
    """Initializes and validates Supabase and Crawler clients."""
    if not all([START_URL, MISTRAL_API_KEY, SUPABASE_URL, SUPABASE_KEY]):
        print(
            "❌ FATAL: Missing one or more environment variables. Please check your .env file."
        )
        return None, None

    try:
        supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("✅ Supabase client initialized.")
    except Exception as e:
        print(f"❌ FATAL: Failed to initialize Supabase client. Error: {e}")
        return None, None

    try:
        crawler_client = AsyncWebCrawler()
        print("✅ crawl4ai client initialized.")
        # Test the crawler
        await crawler_client.start()
        print("✅ Browser initialized successfully.")
    except Exception as e:
        print(f"❌ Failed to initialize crawler: {e}")
        return None, None

    return supabase_client, crawler_client


async def save_to_supabase(supabase: Client, data: dict):
    """Saves a record to the Supabase table, updating if the URL already exists."""
    try:
        response = (
            supabase.table("pages_data").upsert(data, on_conflict="url").execute()
        )
        if response.data:
            print("  [+] Data successfully saved to Supabase.")
        else:
            # Check for specific error details from Supabase response
            error_details = (
                response.error
                if hasattr(response, "error") and response.error
                else "Unknown error"
            )
            print(f"  [!] Supabase save failed. Details: {error_details}")
    except Exception as e:
        print(f"  [!] An exception occurred while saving to Supabase: {e}")


# --- Main Execution Logic ---


async def main():
    """The main function to orchestrate the crawling and data processing workflow."""
    supabase, crawler = None, None  # Initialize to None for the finally block
    try:
        print("Starting crawler setup...")
        supabase, crawler = await initialize_clients()
        if not all([supabase, crawler]):
            return  # Exit if initialization failed
        print("Initialization complete...")

        # LLM configuration is now part of the strategy, not a separate client
        llm_config = {
            "provider": "mistral",
            "model": "mistral-small-latest",
            "api_key": MISTRAL_API_KEY,
        }

        # --- LLM Pre-flight Check ---
        # We make a direct API call to get a clear error message if the key is invalid.
        print("\n[*] Performing pre-flight check on LLM connection...")
        is_valid, message = await check_mistral_api_key(
            MISTRAL_API_KEY, llm_config["model"]
        )
        if not is_valid:
            print(f"❌ LLM pre-flight check failed. Reason: {message}")
            print(
                "   Please verify your MISTRAL_API_KEY in the .env file and your network connection."
            )
            return
        print(f"✅ LLM connection is OK. ({message})")

        urls_to_visit = deque([START_URL])
        visited_urls = set()

        print(f"\n🚀 Starting production crawl at {START_URL}")
        print(f"Will stop after crawling a maximum of {MAX_PAGES_TO_CRAWL} pages.")

        while urls_to_visit and len(visited_urls) < MAX_PAGES_TO_CRAWL:
            current_url = urls_to_visit.popleft()
            if current_url in visited_urls:
                continue

            print(
                f"\n--- Processing page {len(visited_urls) + 1}/{MAX_PAGES_TO_CRAWL} ---"
            )
            print(f"URL: {current_url}")
            visited_urls.add(current_url)

            try:
                # Step 1: Crawl the page to get the raw HTML content.
                crawl_result = await crawler.arun(url=current_url)
                if not crawl_result.success or not crawl_result.cleaned_html:
                    print(
                        f"  [!] Skipping: Page crawl failed or returned no content. Reason: {crawl_result.error_message}"
                    )
                    continue

                # Step 2: Use our robust function to classify the page.
                print("  [*] Classifying page content...")
                classification_prompt = "Analyze the page content and respond with a JSON object with one key: 'page_type'. The value must be one of: Blog, Info, Contact, Other."
                classification_data = await extract_with_llm(
                    api_key=MISTRAL_API_KEY,
                    model=llm_config["model"],
                    prompt_template=classification_prompt,
                    context=crawl_result.cleaned_html,
                    pydantic_model=PageClassification,
                )

                if not classification_data:
                    print("  [!] Skipping page due to classification failure.")
                    continue

                page_type = getattr(classification_data, "page_type", "Other")
                print(f"  [*] LLM classified page as: {page_type}")

                data_for_db = {"url": current_url, "page_type": page_type}

                # Step 3: If it's a blog, run a second, detailed extraction.
                if page_type == "Blog":
                    print("  [*] Extracting blog details...")
                    blog_detail_prompt = "From the page HTML, extract the blog's title, a summary, and the full content. Respond with a JSON object with keys: 'title', 'summary', 'content'."
                    blog_details = await extract_with_llm(
                        api_key=MISTRAL_API_KEY,
                        model=llm_config["model"],
                        prompt_template=blog_detail_prompt,
                        context=crawl_result.cleaned_html,
                        pydantic_model=BlogDetails,
                    )

                    if blog_details:
                        # Convert Pydantic model to dict for Supabase using the modern model_dump()
                        data_for_db.update(blog_details.model_dump())
                        print(
                            f"    -> Extracted Title: {data_for_db.get('title', 'N/A')}"
                        )
                    else:
                        print("    -> Blog detail extraction did not return data.")

                await save_to_supabase(supabase, data_for_db)

                # Discover new URLs to crawl from the initial classification crawl result
                for link in crawl_result.links.get("internal", []):
                    absolute_url = urljoin(current_url, link["href"])
                    clean_url = (
                        urlparse(absolute_url)._replace(query="", fragment="").geturl()
                    )

                    if clean_url not in visited_urls and clean_url not in urls_to_visit:
                        urls_to_visit.append(clean_url)

            except Exception as e:
                print(
                    f"  [!] CRITICAL ERROR in main loop for {current_url}. Error: {e}"
                )
                continue

        print(f"\n✅ Crawl finished. Visited {len(visited_urls)} pages.")

    finally:
        # Ensure the crawler's browser is closed properly to prevent orphaned processes.
        if crawler:
            print("\nShutting down crawler...")
            await crawler.close()
            print("Crawler shut down.")


if __name__ == "__main__":
    asyncio.run(main())
