#!/bin/bash
# doi_or_title_fetcher.sh
# Usage examples:
#   ./doi_or_title_fetcher.sh "10.1145/3511808.3557220"
#   ./doi_or_title_fetcher.sh "Promptagator Few-shot dense retrieval from 8 examples"
#   ./doi_or_title_fetcher.sh "semantic search"  # Short query using Semantic Scholar

set -euo pipefail

########################################
# Helper: API call with retry
########################################
make_api_call() {
  local url="$1"
  local max_retries=3
  local retry_delay=2
  local attempt=1

  while [[ $attempt -le $max_retries ]]; do
    local response
    response=$(curl -s -w "\n%{http_code}" "$url")
    local http_code
    http_code=$(echo "$response" | tail -n1)
    response=$(echo "$response" | sed '$d')

    if [[ "$http_code" == "200" ]]; then
      echo "$response"
      return 0
    elif [[ "$http_code" == "429" ]]; then
      if [[ $attempt -lt $max_retries ]]; then
        echo "⚠️ Rate limited. Waiting ${retry_delay}s before retry..." >&2
        sleep "$retry_delay"
        retry_delay=$((retry_delay * 2))
        attempt=$((attempt + 1))
      else
        echo "❌ Rate limit exceeded after $max_retries attempts" >&2
        return 1
      fi
    else
      echo "❌ API returned status code $http_code" >&2
      return 1
    fi
  done
}

########################################
# Helper: BibTeX formatting
########################################
format_bibtex() {
  local title="$1"
  local authors="$2"
  local year="$3"
  local venue="$4"
  local doi="$5"
  local entry_type="$6"
  local venue_field="$7"

  # Generate citation key
  local first_author
  first_author=$(echo "$authors" | awk -F ' and ' '{print $1}' | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z]//g')
  local first_word
  first_word=$(echo "$title" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z ]//g' | awk '{print $1}')
  local key="${first_author}${year}${first_word}"

  # Build BibTeX entry
  local bib="@${entry_type}{${key},
  title         = {${title}},
  author        = {${authors}},
  year          = {${year}}"
  [[ -n "$venue" && "$venue" != "null" ]] && bib+=",  ${venue_field}     = {${venue}}"
  [[ "$doi" != "null" ]] && bib+=",  doi           = {${doi}}"
  bib+="}"

  echo "$bib"
}

########################################
# Helper: Format and return BibTeX
########################################
format_and_return() {
  local bib="$1"
  local tidied=""
  
  # Try to tidy if bibtex-tidy is available
  if command -v bibtex-tidy >/dev/null 2>&1; then
    tidied=$(echo "$bib" | bibtex-tidy \
               --generate-keys --curly --align=14 --sort-fields --blank-lines \
               --no-escape --drop-all-caps --enclosing-braces \
               --remove-empty-fields --quiet 2>/dev/null || echo "")
  fi

  [[ -z "$tidied" ]] && tidied="$bib"
  echo "$tidied"
}

########################################
# Helper: Generate BibTeX key from title
########################################
generate_bibtex_key() {
  local title="$1"
  local authors="$2"
  local year="$3"

  # Generate a temporary BibTeX entry
  local temp_bib="@article{temp,
  title         = {${title}},
  author        = {${authors}},
  year          = {${year}}
}"

  # Use bibtex-tidy to generate a proper key if available
  if command -v bibtex-tidy >/dev/null 2>&1; then
    local key
    key=$(echo "$temp_bib" | bibtex-tidy --generate-keys --quiet 2>/dev/null | head -1 | sed -E 's/@[^{}]+{//;s/,.*//' || echo "")
    [[ -n "$key" ]] && echo "$key" || echo "temp"
  else
    echo "temp"
  fi
}

########################################
# Helper: BibTeX key similarity check
########################################
check_key_similarity() {
  local query="$1"
  local returned="$2"
  python3 -c 'import sys,difflib;print(int(difflib.SequenceMatcher(None,sys.argv[1].lower(),sys.argv[2].lower()).ratio()*100))' \
    "$query" "$returned"
}

########################################
# Helper: Get BibTeX from DOI
########################################
get_bibtex_from_doi() {
  local doi="$1"
  [[ "$doi" != http* ]] && doi="https://dx.doi.org/${doi#doi:}"
  curl -sL -H "Accept: application/x-bibtex" "$doi"
}

########################################
# OpenAlex search
########################################
search_openalex() {
  local query="$1"
  local encoded
  encoded=$(printf "%s" "$query" | jq -sRr @uri)
  local oa_url="https://api.openalex.org/works?filter=title.search:%22${encoded}%22"

  echo "🔍 Searching OpenAlex..." >&2
  local oa_json
  oa_json=$(make_api_call "$oa_url") || return 1

  if ! echo "$oa_json" | jq -e '.results[0]' >/dev/null 2>&1; then
    echo "❌ No OpenAlex match" >&2
    return 1
  fi

  # Extract fields
  local title
  title=$(echo "$oa_json" | jq -r '.results[0].title')
  local year
  year=$(echo "$oa_json" | jq -r '.results[0].publication_year')
  local venue
  venue=$(echo "$oa_json" | jq -r '.results[0].primary_location.source.display_name')
  local doi
  doi=$(echo "$oa_json" | jq -r '.results[0].doi')
  local authors
  authors=$(echo "$oa_json" | jq -r '.results[0].authorships | map(.author.display_name) | join(" and ")')

  # Check title similarity
  local sim
  sim=$(check_key_similarity "$query" "$title")
  if (( sim < 70 )); then
    echo "⚠️ OpenAlex match similarity too low ($sim%)" >&2
    return 1
  fi

  # First try to get BibTeX from DOI
  if [[ "$doi" != "null" ]]; then
    echo "🔍 Trying to fetch BibTeX from DOI..." >&2
    local bib
    bib=$(get_bibtex_from_doi "$doi")
    if [[ -n "$bib" ]]; then
      format_and_return "$bib"
      return 0
    fi
    echo "⚠️ Could not fetch BibTeX from DOI, falling back to OpenAlex data..." >&2
  fi

  # If DOI fetch failed, construct BibTeX from OpenAlex data
  local entry_type="article"
  local venue_field="journal"
  [[ "$venue" =~ (Conference|Workshop|Symposium) ]] && { entry_type="inproceedings"; venue_field="booktitle"; }

  local bib
  bib=$(format_bibtex "$title" "$authors" "$year" "$venue" "$doi" "$entry_type" "$venue_field")
  format_and_return "$bib"
}

########################################
# Semantic Scholar search
########################################
search_semantic_scholar() {
  local query_key="$1"
  local query=$(echo "$query_key" | sed -E 's/([a-z]+)([0-9]{4})(.*)/\1 \2 \3/')
  local encoded
  encoded=$(printf "%s" "$query" | jq -sRr @uri)
  local ss_url="https://api.semanticscholar.org/graph/v1/paper/search?query=${encoded}&sort=relevance"
  echo "$ss_url" >&2
  echo "🔍 Searching Semantic Scholar..." >&2
  local ss_json
  ss_json=$(make_api_call "$ss_url") || return 1

  if ! echo "$ss_json" | jq -e '.data[0]' >/dev/null 2>&1; then
    echo "❌ No Semantic Scholar match" >&2
    return 1
  fi

  # Get the paper ID and title from the first result
  local paper_id
  paper_id=$(echo "$ss_json" | jq -r '.data[0].paperId')
  local paper_title
  paper_title=$(echo "$ss_json" | jq -r '.data[0].title')

  # Fetch detailed paper information using the paper ID
  local paper_url="https://api.semanticscholar.org/graph/v1/paper/${paper_id}?fields=externalIds,year,venue,authors"
  echo "🔍 Fetching paper details..." >&2
  local paper_json
  paper_json=$(make_api_call "$paper_url") || return 1

  # Extract DOI from externalIds
  local doi
  doi=$(echo "$paper_json" | jq -r '.externalIds.DOI')

  # Extract other fields
  local year
  year=$(echo "$paper_json" | jq -r '.year')
  local venue
  venue=$(echo "$paper_json" | jq -r '.venue')
  local authors
  authors=$(echo "$paper_json" | jq -r '.authors | map(.name) | join(" and ")')

  # Generate BibTeX keys for comparison
  local match_key
  match_key=$(generate_bibtex_key "$paper_title" "$authors" "$year")

  # Check key similarity
  local sim
  echo "Query key: $query_key" >&2
  echo "Match key: $match_key" >&2
  sim=$(check_key_similarity "$query_key" "$match_key")
  if (( sim < 70 )); then
    echo "⚠️ Semantic Scholar match key similarity too low ($sim%)" >&2
    return 1
  fi

  # First try to get BibTeX from DOI
  if [[ "$doi" != "null" ]]; then
    echo "🔍 Trying to fetch BibTeX from DOI..." >&2
    local bib
    bib=$(get_bibtex_from_doi "$doi")
    if [[ -n "$bib" ]]; then
      format_and_return "$bib"
      return 0
    fi
    echo "⚠️ Could not fetch BibTeX from DOI, falling back to Semantic Scholar data..." >&2
  fi

  # If DOI fetch failed, construct BibTeX from Semantic Scholar data
  local entry_type="article"
  local venue_field="journal"
  [[ "$venue" =~ (Conference|Workshop|Symposium) ]] && { entry_type="inproceedings"; venue_field="booktitle"; }

  local bib
  bib=$(format_bibtex "$paper_title" "$authors" "$year" "$venue" "$doi" "$entry_type" "$venue_field")
  format_and_return "$bib"
}

########################################
# Main function
########################################
main() {
  echo "🔍 Starting DOI/Title fetcher…" >&2
  echo "Input: $1" >&2

  local raw="$1"
  raw="$(echo "$raw" | tr -d '[:space:]')"
  local is_doi=0
  [[ "$raw" == http* || "$raw" == doi:* || "$raw" == 10.*/* ]] && is_doi=1

  if [[ $is_doi -eq 0 ]]; then
    # Count words in the query
    local word_count
    word_count=$(echo "$1" | wc -w)

    if [[ $word_count -lt 4 ]]; then
      # Short query: try Semantic Scholar
      search_semantic_scholar "$1" || search_openalex "$1"
    else
      # Long query: try OpenAlex
      search_openalex "$1"
    fi
  else
    # DOI search
    local doi="$raw"
    echo "🔍 Fetching BibTeX from DOI..." >&2
    local bib
    bib=$(get_bibtex_from_doi "$doi") || { echo "❌ Could not fetch BibTeX" >&2; exit 1; }
    format_and_return "$bib"
  fi
}

# Run main function
main "$1"
