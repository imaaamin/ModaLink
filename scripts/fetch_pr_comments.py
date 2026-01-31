"""Script to fetch PR comments from a public GitHub repository."""

import sys
import json
import requests
from typing import List, Dict, Any
from pathlib import Path

# Add parent directory to path if needed for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def fetch_pr_comments(owner: str, repo: str, pr_number: int) -> List[Dict[str, Any]]:
    """
    Fetch all comments from a GitHub PR.
    
    Args:
        owner: Repository owner (e.g., 'octocat')
        repo: Repository name (e.g., 'Hello-World')
        pr_number: PR number
    
    Returns:
        List of comment dictionaries
    """
    base_url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}"
    
    # Fetch PR details
    print(f"Fetching PR #{pr_number} from {owner}/{repo}...")
    pr_response = requests.get(base_url)
    pr_response.raise_for_status()
    pr_data = pr_response.json()
    
    print(f"PR Title: {pr_data.get('title', 'N/A')}")
    print(f"PR State: {pr_data.get('state', 'N/A')}")
    print(f"Author: {pr_data.get('user', {}).get('login', 'N/A')}")
    print()
    
    all_comments = []
    
    # Fetch review comments (comments on code)
    review_comments_url = f"{base_url}/comments"
    print("Fetching review comments...")
    review_response = requests.get(review_comments_url)
    review_response.raise_for_status()
    review_comments = review_response.json()
    all_comments.extend(review_comments)
    print(f"Found {len(review_comments)} review comments")
    
    # Fetch issue comments (general PR comments)
    issue_comments_url = f"https://api.github.com/repos/{owner}/{repo}/issues/{pr_number}/comments"
    print("Fetching issue comments...")
    issue_response = requests.get(issue_comments_url)
    issue_response.raise_for_status()
    issue_comments = issue_response.json()
    all_comments.extend(issue_comments)
    print(f"Found {len(issue_comments)} issue comments")
    
    # Fetch review comments (from reviews)
    reviews_url = f"{base_url}/reviews"
    print("Fetching review comments from reviews...")
    reviews_response = requests.get(reviews_url)
    reviews_response.raise_for_status()
    reviews = reviews_response.json()
    
    for review in reviews:
        if review.get('body'):
            all_comments.append({
                'type': 'review',
                'user': review.get('user', {}),
                'body': review.get('body'),
                'state': review.get('state'),
                'created_at': review.get('submitted_at') or review.get('created_at'),
                'html_url': review.get('html_url')
            })
    
    print(f"Found {len(reviews)} reviews")
    print(f"\nTotal comments: {len(all_comments)}")
    
    return all_comments


def print_comments(comments: List[Dict[str, Any]]):
    """Print comments in a readable format."""
    for i, comment in enumerate(comments, 1):
        print(f"\n{'='*80}")
        print(f"Comment #{i}")
        print(f"{'='*80}")
        
        user = comment.get('user', {}).get('login', 'Unknown')
        created_at = comment.get('created_at', comment.get('submitted_at', 'N/A'))
        comment_type = comment.get('type', 'comment')
        
        print(f"Author: {user}")
        print(f"Type: {comment_type}")
        print(f"Created: {created_at}")
        
        if comment.get('path'):
            print(f"File: {comment.get('path')}")
            print(f"Line: {comment.get('line', 'N/A')}")
        
        if comment.get('state'):
            print(f"State: {comment.get('state')}")
        
        print(f"\nBody:")
        print(comment.get('body', 'No body'))
        
        if comment.get('html_url'):
            print(f"\nURL: {comment.get('html_url')}")


def main():
    """Main function."""
    if len(sys.argv) < 4:
        print("Usage: python fetch_pr_comments.py <owner> <repo> <pr_number> [output_file]")
        print("\nExample:")
        print("  python fetch_pr_comments.py microsoft vscode 12345")
        print("  python fetch_pr_comments.py microsoft vscode 12345 comments.json")
        sys.exit(1)
    
    owner = sys.argv[1]
    repo = sys.argv[2]
    try:
        pr_number = int(sys.argv[3])
    except ValueError:
        print(f"Error: PR number must be an integer, got: {sys.argv[3]}")
        sys.exit(1)
    
    output_file = sys.argv[4] if len(sys.argv) > 4 else None
    
    try:
        comments = fetch_pr_comments(owner, repo, pr_number)
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(comments, f, indent=2, ensure_ascii=False)
            print(f"\n✓ Comments saved to {output_file}")
        else:
            print_comments(comments)
    
    except requests.exceptions.HTTPError as e:
        print(f"Error fetching PR: {e}")
        if e.response.status_code == 404:
            print("PR not found. Make sure the repository is public and the PR number is correct.")
        elif e.response.status_code == 403:
            print("Rate limit exceeded or repository is private. Try again later or use authentication.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
