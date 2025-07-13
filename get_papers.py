from semanticscholar import SemanticScholar
from semanticscholar.SemanticScholarException import NoMorePagesException
import json
import os

# --- Constants and Configuration ---

# The query for fetching papers from Semantic Scholar. You can update this.
SEARCH_QUERY = (
    '("rolling stock" | "rolling-stock" | "high speed train" | "high-speed train" | "railway vehicle" | locomotive | railcar | bogie | pantograph | maglev | tramway | metro | "light rail")'
    '+'
    '('
        'dynamics | vibration | stability | suspension | mechatronics | "control system" |'
        'maintenance | diagnostics | "condition monitoring" | "fault detection" | prognostics | "remaining useful life" | RUL |'
        'traction | propulsion | "brak*" | "regenerative braking" '
        'aerodynamics | "crosswind" | "structural design" | "crashworthiness" | fatigue |'
        '("wheel" ~3 "rail") | ("pantograph" ~3 "catenary") |'
        'energy | hydrogen | "fuel cell" | battery | hybrid | "power module" | circuit |'
        'noise | acoustics | HVAC |'
        '"signalling" | ERTMS | CBTC |'
        'axle | "bearing*" | coupler | "gangway connection"'
    ')'
)

# Define the output file and save frequency
OUTPUT_FILENAME = "papers_dataset.json"
SAVE_INTERVAL = 10  # Save after this many pages are processed
MAX_PAGES_TO_FETCH = 200 # Safety limit for the number of pages to fetch in a single run

# --- Helper Functions ---

def load_existing_data(filename: str) -> dict:
    """Loads existing papers from the JSON file into a dictionary for quick lookup."""
    if not os.path.exists(filename):
        print(f"'{filename}' not found. A new file will be created.")
        return {}

    try:
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Create a dictionary with paperId as the key for efficient duplicate checking
            return {item['paperId']: item for item in data}
    except (json.JSONDecodeError, IOError):
        print(f"Warning: Could not read or parse '{filename}'. Starting with an empty dataset.")
        return {}

def save_data(filename: str, papers_dict: dict):
    """Saves the collected paper data to the JSON file."""
    # Convert the dictionary values back to a list for JSON serialization
    papers_list = list(papers_dict.values())
    with open(filename, "w+", encoding="utf-8") as f:
        json.dump(papers_list, f, indent=4)
    print(f"💾 Successfully saved {len(papers_list)} papers to '{filename}'.")

# --- Main Script ---

def main():
    """Main function to fetch, process, and save papers."""
    
    # 1. Load existing data to avoid duplicates
    print("Loading existing dataset...")
    papers_data = load_existing_data(OUTPUT_FILENAME)
    initial_count = len(papers_data)
    print(f"Found {initial_count} existing papers.")

    # 2. Initialize Semantic Scholar client and search
    sch = SemanticScholar(timeout=15, debug=False, retry=True)
    
    print(f"\nSearching for new papers with the specified query...")
    search_results = sch.search_paper(
        query=SEARCH_QUERY,
        fields=['paperId', 'title', 'abstract', 'url', 'year'],
        fields_of_study=["Engineering"],
        year="2000-",
        bulk=True
    )

    # 3. Iterate through pages, collect new data, and save incrementally
    pages_processed = 0
    new_papers_found_this_run = 0
    papers = search_results.items
    earlier_size = 1

    while pages_processed < MAX_PAGES_TO_FETCH:
        # Process items on the current page
        for item in papers[earlier_size-1:]:
            # Skip if abstract is missing or if paper is already in our dataset
            if item["abstract"] is None or item['paperId'] in papers_data:
                continue

            # Add new paper to our data dictionary
            paper_id = item['paperId']
            papers_data[paper_id] = {
                "paperId": paper_id,
                "title": item['title'],
                "abstract": item['abstract'],
                "year": item['year'],
                "url": item['url']
            }
            new_papers_found_this_run += 1

        print(f"📄 Page {pages_processed + 1} processed. Total papers in memory: {len(papers_data)}.")

        pages_processed += 1
        
        # Save results at the specified interval
        if pages_processed % SAVE_INTERVAL == 0:
            print(f"\n--- Reached save interval of {SAVE_INTERVAL} pages ---")
            save_data(OUTPUT_FILENAME, papers_data)

        # Try to fetch the next page of results
        try:
            earlier_size = len(search_results.items)
            search_results.next_page()
        except NoMorePagesException:
            print("\nNo more pages left in search results.")
            break
    
    # 4. Final save and summary
    print("\n--- Search finished or page limit reached ---")
    if len(papers_data) > initial_count:
        save_data(OUTPUT_FILENAME, papers_data)
    else:
        print("No new papers were added in this run.")
        
    print("\n--- Summary ---")
    print(f"Initial paper count: {initial_count}")
    print(f"New papers found in this run: {new_papers_found_this_run}")
    print(f"Total papers in dataset: {len(papers_data)}")
    print("✅ Script finished.")


if __name__ == "__main__":
    main()