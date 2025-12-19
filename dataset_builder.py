"""
Dataset Builder for Manga Vectorizer

Automatically downloads manga panels to create datasets for clustering experiments.
Creates three dataset sizes:
- Small: 9 mangas (3 authors × 3 works)
- Medium: 21 mangas (7 authors × 3 works)
- Large: 51 mangas (17 authors × 3 works)

Each manga will have 20 panels downloaded from multiple chapters.
"""

import requests
import json
import random
import time
from pathlib import Path
from dataclasses import dataclass, field

# Import from manga_downloader
from manga_downloader import (
    BASE_URL,
    MANGA_URL,
    CHAPTER_URL,
    PAGE_URL,
    HEADERS,
    get_chapter_pages,
    select_human_name,
)


# Dataset output directories
DATASET_DIR = Path("./datasets")

# Configuration
PANELS_PER_MANGA = 20
WORKS_PER_AUTHOR = 3
MIN_CHAPTERS_PER_MANGA = 5   # Need enough chapters to spread pages across
PAGES_TO_SKIP_START = 5      # Skip first N pages (ads/title/cover)
PAGES_TO_SKIP_END = 5        # Skip last N pages (ads/credits)
MIN_PAGES_PER_CHAPTER = 12   # Chapters need 12+ pages to have valid middle pages


@dataclass
class MangaWork:
    """Represents a manga work with its metadata."""
    id: str
    title: str
    author_id: str
    author_name: str
    status: str
    content_rating: str
    chapters: list = field(default_factory=list)
    valid_pages_count: int = 0


@dataclass
class Author:
    """Represents an author with their works."""
    id: str
    name: str
    works: list[MangaWork] = field(default_factory=list)


def rate_limit_sleep(seconds: float = 0.3):
    """Sleep to avoid rate limiting."""
    time.sleep(seconds)


def fetch_author_works(author_id: str) -> tuple[str, list[dict]]:
    """
    Fetch author name and all their manga works from API.
    Returns (author_name, list of manga dicts).
    """
    author_name = "Unknown"
    works = []
    
    try:
        # Get author info
        resp = requests.get(f"{BASE_URL}/author/{author_id}", headers=HEADERS)
        rate_limit_sleep(0.2)
        data = resp.json()
        
        if data.get("data"):
            author_name = data["data"]["attributes"].get("name", "Unknown")
        
        # Get all manga by this author
        params = {
            "authors[]": [author_id],
            "limit": 100,
            "availableTranslatedLanguage[]": ["en"],
            "hasAvailableChapters": "true",
            "contentRating[]": ["safe", "suggestive"],
        }
        
        manga_resp = requests.get(MANGA_URL, params=params, headers=HEADERS)
        rate_limit_sleep(0.2)
        manga_data = manga_resp.json()
        
        for manga in manga_data.get("data", []):
            # Skip manga with official retail links
            links = manga["attributes"].get("links", {}) or {}
            if any(k in links for k in ["amazon", "ebj", "cdj"]):
                continue
            
            titles = manga["attributes"]["title"]
            title = titles.get("en") or titles.get("ja-ro") or list(titles.values())[0] if titles else "Unknown"
            
            # Skip colored manga (we want black & white only)
            title_lower = title.lower()
            if any(term in title_lower for term in ["colored", "color", "colour", "coloured", "full color"]):
                continue
            
            works.append({
                "id": manga["id"],
                "title": title,
                "status": manga["attributes"].get("status", "unknown"),
                "content_rating": manga["attributes"].get("contentRating", "unknown"),
            })
    except Exception as e:
        print(f"      Error fetching author {author_id}: {e}")
    
    return author_name, works


def search_authors_with_works(min_works: int = 3, limit: int = 100) -> list[Author]:
    """
    Search for authors who have at least min_works different manga.
    Strategy: Find popular manga, then fetch ALL works by each unique author.
    """
    print(f"\n🔍 Searching for authors with at least {min_works} works...")
    
    # Step 1: Find unique author IDs from multiple manga searches
    print("   Step 1: Finding authors from popular manga (multiple pages)...")
    
    discovered_author_ids = set()
    
    # Search with different orderings and offsets to find more authors
    search_configs = [
        {"order[followedCount]": "desc", "offset": 0},
        {"order[followedCount]": "desc", "offset": 100},
        {"order[followedCount]": "desc", "offset": 200},
        {"order[rating]": "desc", "offset": 0},
        {"order[rating]": "desc", "offset": 100},
        {"order[createdAt]": "desc", "offset": 0},
    ]
    
    for config in search_configs:
        params = {
            "limit": 100,
            "includes[]": ["author"],
            "availableTranslatedLanguage[]": ["en"],
            "hasAvailableChapters": "true",
            "contentRating[]": ["safe", "suggestive"],
            **config
        }
        
        try:
            resp = requests.get(MANGA_URL, params=params, headers=HEADERS)
            rate_limit_sleep()
            data = resp.json()
            
            for manga in data.get("data", []):
                for rel in manga.get("relationships", []):
                    if rel["type"] == "author":
                        discovered_author_ids.add(rel["id"])
                        break
        except Exception as e:
            print(f"      Warning: Search failed - {e}")
            continue
        
        print(f"      Found {len(discovered_author_ids)} unique authors so far...")
    
    print(f"      Total: {len(discovered_author_ids)} unique authors discovered")
    
    # Step 2: For each author, fetch ALL their works
    print(f"   Step 2: Fetching works for each author...")
    
    qualified_authors = []
    checked_count = 0
    
    for author_id in discovered_author_ids:
        checked_count += 1
        
        author_name, works = fetch_author_works(author_id)
        
        if author_name == "Unknown" or len(works) < min_works:
            continue
        
        # Create Author with MangaWork objects
        author = Author(id=author_id, name=author_name)
        for w in works:
            author.works.append(MangaWork(
                id=w["id"],
                title=w["title"],
                author_id=author_id,
                author_name=author_name,
                status=w["status"],
                content_rating=w["content_rating"],
            ))
        
        qualified_authors.append(author)
        print(f"      ✓ {author_name}: {len(works)} works")
        
        if len(qualified_authors) >= limit:
            break
        
        # Progress update
        if checked_count % 10 == 0:
            print(f"      Checked {checked_count}/{len(discovered_author_ids)} authors, found {len(qualified_authors)} qualified")
    
    print(f"\n   ✓ Found {len(qualified_authors)} authors with {min_works}+ works")
    return qualified_authors


def get_valid_chapters(manga_id: str) -> list[dict]:
    """
    Get chapters for a manga, filtering out:
    - Float chapters (bonus/special chapters)
    - Chapters with too few pages
    Returns list of valid chapter info dicts.
    """
    # Get aggregate chapter info
    resp = requests.get(
        f"{MANGA_URL}/{manga_id}/aggregate",
        params={"translatedLanguage[]": ["en"]},
        headers=HEADERS
    )
    rate_limit_sleep()
    
    data = resp.json()
    chapters = []
    
    volumes = data.get("volumes", {})
    
    # Handle case where volumes is empty list instead of dict
    if not volumes or not isinstance(volumes, dict):
        return chapters
    
    for vol_key, vol_data in volumes.items():
        vol_chapters = vol_data.get("chapters", {})
        
        # Handle case where chapters is a list instead of dict
        if not vol_chapters or not isinstance(vol_chapters, dict):
            continue
        
        for ch_key, ch_data in vol_chapters.items():
            try:
                # Skip non-numeric or float chapters
                if ch_key == "none":
                    continue
                ch_num = float(ch_key)
                # Skip float chapters (like 5.5, 10.1 - usually bonus)
                if ch_num != int(ch_num):
                    continue
                
                chapters.append({
                    "chapter": ch_key,
                    "chapter_num": int(ch_num),
                    "id": ch_data.get("id"),
                })
            except (ValueError, AttributeError):
                continue
    
    # Sort by chapter number
    chapters.sort(key=lambda x: x["chapter_num"])
    return chapters


def validate_manga_for_dataset(work: MangaWork) -> bool:
    """
    Validate that a manga has enough chapters AND enough valid pages.
    Samples chapters to estimate total valid pages available.
    Requires conservative estimate to ensure we can actually get PANELS_PER_MANGA pages.
    """
    chapters = get_valid_chapters(work.id)
    
    if len(chapters) < MIN_CHAPTERS_PER_MANGA:
        print(f"❌ Only {len(chapters)} chapters (need {MIN_CHAPTERS_PER_MANGA}+)")
        return False
    
    # Sample more chapters for better estimate
    sample_size = min(8, len(chapters))
    sample_chapters = random.sample(chapters, sample_size)
    
    total_valid_pages = 0
    chapters_with_valid_pages = 0
    failed_chapters = 0
    
    for ch in sample_chapters:
        rate_limit_sleep(0.25)
        chapter_data = get_chapter_pages(ch["id"], quiet=True)
        
        if not chapter_data:
            failed_chapters += 1
            continue
        
        total_pages = chapter_data["total"]
        valid_pages = total_pages - PAGES_TO_SKIP_START - PAGES_TO_SKIP_END
        
        if valid_pages >= 2:  # At least 2 valid pages in middle
            total_valid_pages += valid_pages
            chapters_with_valid_pages += 1
    
    # If more than half of sampled chapters failed, reject
    if failed_chapters > sample_size // 2:
        print(f"❌ Too many unavailable chapters ({failed_chapters}/{sample_size})")
        return False
    
    # Estimate total valid pages across all chapters
    if chapters_with_valid_pages == 0:
        print(f"❌ No chapters with valid middle pages")
        return False
    
    avg_valid_pages = total_valid_pages / chapters_with_valid_pages
    # Conservative estimate: assume some chapters will fail
    success_rate = chapters_with_valid_pages / sample_size
    estimated_total = avg_valid_pages * len(chapters) * success_rate
    
    # Need at least 1.5x PANELS_PER_MANGA to be safe
    required_pages = PANELS_PER_MANGA * 1.5
    
    if estimated_total < required_pages:
        print(f"❌ ~{int(estimated_total)} est. pages (need {int(required_pages)}+ for safety)")
        return False
    
    work.chapters = chapters
    work.valid_pages_count = int(estimated_total)
    print(f"✓ {len(chapters)} ch, ~{int(estimated_total)} pages (success rate: {success_rate:.0%})")
    return True


def select_authors_for_dataset(authors: list[Author], num_authors: int) -> list[Author]:
    """
    Select authors for the dataset, validating their works.
    Ensures each author has exactly WORKS_PER_AUTHOR valid manga.
    Returns list of authors with validated works.
    """
    print(f"\n📋 Selecting {num_authors} authors (each with {WORKS_PER_AUTHOR} manga)...")
    
    selected = []
    random.shuffle(authors)
    
    for author in authors:
        if len(selected) >= num_authors:
            break
        
        print(f"\n   Checking author: {author.name} ({len(author.works)} works available)")
        
        # Validate works for this author - check ALL works if needed
        valid_works = []
        for work in author.works:
            if validate_manga_for_dataset(work):
                valid_works.append(work)
                print(f"      ✓ Valid: {work.title[:40]}")
            else:
                print(f"      ✗ Invalid: {work.title[:40]}")
            
            if len(valid_works) >= WORKS_PER_AUTHOR:
                break
            
            rate_limit_sleep(0.3)
        
        if len(valid_works) >= WORKS_PER_AUTHOR:
            author.works = valid_works[:WORKS_PER_AUTHOR]
            selected.append(author)
            print(f"   ✅ Selected {author.name} with {WORKS_PER_AUTHOR} works")
        else:
            print(f"   ❌ {author.name} only has {len(valid_works)}/{WORKS_PER_AUTHOR} valid works - SKIPPING")
    
    if len(selected) < num_authors:
        print(f"\n   ⚠️  WARNING: Only found {len(selected)}/{num_authors} qualified authors!")
    
    return selected


def download_pages_for_manga(
    work: MangaWork,
    output_dir: Path,
    num_pages: int = PANELS_PER_MANGA
) -> int:
    """
    Download pages for a manga, spreading across multiple chapters.
    Returns number of pages actually downloaded.
    """
    safe_author = "".join(c for c in work.author_name if c.isalnum() or c in (' ', '-', '_')).strip()
    safe_title = "".join(c for c in work.title if c.isalnum() or c in (' ', '-', '_')).strip()
    
    manga_dir = output_dir / safe_author / safe_title
    manga_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metadata
    metadata = {
        "id": work.id,
        "title": work.title,
        "author": work.author_name,
        "author_id": work.author_id,
        "status": work.status,
        "content_rating": work.content_rating,
    }
    with open(manga_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    # Get fresh chapter list
    chapters = get_valid_chapters(work.id)
    if not chapters:
        return 0
    
    # Strategy: spread pages across multiple RANDOM chapters
    # e.g., 20 pages from 5-7 chapters = ~3-4 pages per chapter
    num_chapters_to_use = min(len(chapters), 7)  # Use up to 7 chapters for variety
    pages_per_chapter = max(2, num_pages // num_chapters_to_use)  # At least 2 pages each
    
    # Randomly select chapters from the available pool
    random.shuffle(chapters)
    
    downloaded_count = 0
    chapter_idx = 0
    
    # Keep downloading from random chapters until we have enough pages
    while downloaded_count < num_pages and chapter_idx < len(chapters):
        chapter = chapters[chapter_idx]
        chapter_idx += 1
        
        # How many more pages do we need?
        remaining = num_pages - downloaded_count
        # Get 2-4 pages per chapter to spread across chapters
        pages_to_get = min(remaining, random.randint(2, 4))
        
        rate_limit_sleep(0.3)
        chapter_data = get_chapter_pages(chapter["id"], quiet=True)
        
        if not chapter_data:
            continue  # Silently skip unavailable chapters
        
        pages = chapter_data["pages"]
        total = chapter_data["total"]
        base_url = chapter_data["base_url"]
        chapter_hash = chapter_data["hash"]
        
        # Get valid page indices (skip first and last N)
        valid_start = PAGES_TO_SKIP_START
        valid_end = total - PAGES_TO_SKIP_END
        
        if valid_end <= valid_start:
            continue
        
        valid_indices = list(range(valid_start, valid_end))
        
        if not valid_indices:
            continue
        
        # Select random pages from valid range
        pages_to_get = min(pages_to_get, len(valid_indices))
        selected_indices = sorted(random.sample(valid_indices, pages_to_get))
        
        # Create chapter directory
        chapter_dir = manga_dir / f"chapter_{chapter['chapter']}"
        chapter_dir.mkdir(parents=True, exist_ok=True)
        
        # Download pages
        for page_idx in selected_indices:
            page_filename = pages[page_idx]
            image_url = f"{base_url}/data/{chapter_hash}/{page_filename}"
            
            try:
                rate_limit_sleep(0.2)
                img_resp = requests.get(image_url, headers=HEADERS)
                img_resp.raise_for_status()
                
                ext = Path(page_filename).suffix or ".jpg"
                filename = f"page_{page_idx + 1:03d}{ext}"
                filepath = chapter_dir / filename
                
                with open(filepath, "wb") as f:
                    f.write(img_resp.content)
                
                downloaded_count += 1
                
            except Exception as e:
                print(f"      Error downloading page: {e}")
    
    return downloaded_count


def build_dataset(
    authors: list[Author],
    dataset_name: str,
    output_dir: Path
) -> dict:
    """
    Build a dataset by downloading pages for all works from selected authors.
    Ensures each manga gets exactly PANELS_PER_MANGA pages.
    """
    print(f"\n{'='*60}")
    print(f"   Building Dataset: {dataset_name}")
    print(f"   Target: {len(authors)} authors × {WORKS_PER_AUTHOR} manga × {PANELS_PER_MANGA} pages")
    print(f"{'='*60}")
    
    dataset_dir = output_dir / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    stats = {
        "name": dataset_name,
        "authors": [],
        "total_mangas": 0,
        "total_pages": 0,
        "complete_mangas": 0,  # Mangas with full 20 pages
        "incomplete_mangas": [],  # Track which manga didn't get enough
    }
    
    for author in authors:
        print(f"\n📚 Processing author: {author.name}")
        
        author_stats = {
            "name": author.name,
            "id": author.id,
            "works": []
        }
        
        successful_works = 0
        
        for work in author.works:
            print(f"   📖 Downloading: {work.title[:50]}...")
            
            pages_downloaded = download_pages_for_manga(work, dataset_dir)
            
            if pages_downloaded >= PANELS_PER_MANGA:
                print(f"      ✅ Downloaded {pages_downloaded} pages")
                stats["complete_mangas"] += 1
                successful_works += 1
            else:
                print(f"      ❌ Only got {pages_downloaded}/{PANELS_PER_MANGA} pages - INCOMPLETE")
                stats["incomplete_mangas"].append({
                    "author": author.name,
                    "title": work.title,
                    "pages": pages_downloaded
                })
            
            author_stats["works"].append({
                "title": work.title,
                "id": work.id,
                "pages_downloaded": pages_downloaded
            })
            
            stats["total_pages"] += pages_downloaded
            stats["total_mangas"] += 1
        
        if successful_works == WORKS_PER_AUTHOR:
            print(f"   ✅ {author.name}: {successful_works}/{WORKS_PER_AUTHOR} manga complete")
        else:
            print(f"   ⚠️  {author.name}: Only {successful_works}/{WORKS_PER_AUTHOR} manga complete")
        
        stats["authors"].append(author_stats)
    
    # Save dataset manifest
    manifest_path = dataset_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    # Final summary
    expected_mangas = len(authors) * WORKS_PER_AUTHOR
    expected_pages = expected_mangas * PANELS_PER_MANGA
    
    print(f"\n{'='*60}")
    print(f"   Dataset '{dataset_name}' Summary")
    print(f"{'='*60}")
    print(f"   Authors:    {len(authors)}")
    print(f"   Mangas:     {stats['total_mangas']} (target: {expected_mangas})")
    print(f"   Complete:   {stats['complete_mangas']}/{stats['total_mangas']} manga with {PANELS_PER_MANGA} pages")
    print(f"   Pages:      {stats['total_pages']} (target: {expected_pages})")
    
    if stats["incomplete_mangas"]:
        print(f"\n   ⚠️  Incomplete manga:")
        for m in stats["incomplete_mangas"]:
            print(f"      - {m['author']}/{m['title']}: {m['pages']} pages")
    
    if stats['complete_mangas'] == expected_mangas:
        print(f"\n   ✅ DATASET COMPLETE!")
    else:
        print(f"\n   ❌ DATASET INCOMPLETE - {expected_mangas - stats['complete_mangas']} manga missing full pages")
    
    print(f"\n   Saved to: {dataset_dir}")
    
    return stats


def main():
    """Main entry point for dataset building."""
    print("\n" + "=" * 60)
    print("       MANGA DATASET BUILDER")
    print("=" * 60)
    print(f"\nThis will create three datasets:")
    print(f"  • Small:  9 mangas  (3 authors × 3 works)")
    print(f"  • Medium: 21 mangas (7 authors × 3 works)")
    print(f"  • Large:  51 mangas (17 authors × 3 works)")
    print(f"\nEach manga will have {PANELS_PER_MANGA} panels from multiple chapters.")
    print(f"Pages from first/last {PAGES_TO_SKIP_START} of each chapter are skipped.\n")
    
    # Configuration for datasets
    DATASETS = [
        {"name": "small", "num_authors": 3},
        {"name": "medium", "num_authors": 7},
        {"name": "large", "num_authors": 17},
    ]
    
    # Find authors with multiple works
    all_authors = search_authors_with_works(min_works=WORKS_PER_AUTHOR, limit=100)
    
    if len(all_authors) < 17:
        print(f"\n⚠️  Only found {len(all_authors)} qualified authors.")
        print("   Some datasets may be smaller than expected.")
    
    # Menu for dataset selection
    while True:
        print("\n" + "-" * 40)
        print("[Dataset Builder Menu]")
        print("  1. Build small dataset (3 authors)")
        print("  2. Build medium dataset (7 authors)")
        print("  3. Build large dataset (17 authors)")
        print("  4. Build all datasets")
        print("  5. Show qualified authors")
        print("  0. Exit")
        
        choice = input("\nSelect option: ").strip()
        
        if choice == "0":
            print("\nGoodbye!")
            break
        
        elif choice == "5":
            print(f"\n📋 Qualified Authors ({len(all_authors)} total):")
            for i, author in enumerate(all_authors[:30], 1):
                works_str = ", ".join(w.title[:25] for w in author.works[:3])
                print(f"  {i:2}. {author.name}: {works_str}...")
            continue
        
        elif choice in ["1", "2", "3", "4"]:
            datasets_to_build = []
            
            if choice == "1":
                datasets_to_build = [DATASETS[0]]
            elif choice == "2":
                datasets_to_build = [DATASETS[1]]
            elif choice == "3":
                datasets_to_build = [DATASETS[2]]
            elif choice == "4":
                datasets_to_build = DATASETS
            
            # Calculate total authors needed
            max_authors_needed = max(d["num_authors"] for d in datasets_to_build)
            
            # Select and validate authors
            selected_authors = select_authors_for_dataset(all_authors, max_authors_needed)
            
            if len(selected_authors) < max_authors_needed:
                print(f"\n⚠️  Could only validate {len(selected_authors)} authors.")
                confirm = input("Continue with available authors? (y/n): ").strip().lower()
                if confirm != "y":
                    continue
            
            # Build each dataset
            for ds_config in datasets_to_build:
                num_authors = min(ds_config["num_authors"], len(selected_authors))
                authors_for_ds = selected_authors[:num_authors]
                
                build_dataset(
                    authors=authors_for_ds,
                    dataset_name=ds_config["name"],
                    output_dir=DATASET_DIR
                )
            
            print("\n" + "=" * 60)
            print("   ALL DATASETS COMPLETE!")
            print("=" * 60)
        
        else:
            print("Invalid option.")


if __name__ == "__main__":
    main()
