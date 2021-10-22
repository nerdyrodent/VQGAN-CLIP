"""
Use the ACMI Public API to get an object's metadata
# which is then used to generate images.

# If you'd like to use an individual ACMI `work_id`
# specify it below, else we'll choose a random Work.

# Hint: you can find the ACMI `work_id` in the URL from a page on our website.

# e.g. Untitled Goose Game is ID `118201`
# https://www.acmi.net.au/works/118201--untitled-goose-game/
"""

import random

import subprocess
import requests

from bs4 import BeautifulSoup

works = []
WORK_ID = ''

if WORK_ID:
    # Get a single work if the ID is set
    work = requests.get(f'https://api.acmi.net.au/works/{WORK_ID}/').json()
else:
    # Let's get 50 works (5 pages of results)
    for page in range(1, 6):
        print(f'Getting page {page} of ACMI Works...')
        response = requests.get(
            f'https://api.acmi.net.au/works/?page={page + 1}',
        ).json()
        works += response.get('results', [])

    # Select a random work
    print(f'Selecting a random work from {len(works)} ACMI works...')
    work = random.choice(works)


def clean(input_string):
    """
    Strip all problematic characters and return a clean string.
    """
    # Convert html to plain text
    soup = BeautifulSoup(input_string)
    output_string = soup.get_text()

    # Remove extra spaces
    output_string = output_string.strip()

    # Remove troublesome characters for this Notebook
    bad_characters = [
        ':',
        '|',
        '\n',
        '[',
        ']',
    ]
    for character in bad_characters:
        output_string = output_string.replace(character, '')

    return output_string


# Use the title and brief description fields from our work
# as input for this Notebook
work_metadata_strings = []
if work['title']:
    work_metadata_strings.append(
        clean(work['title'])
    )
if work['brief_description']:
    clean_brief_description = clean(work['brief_description'])
    # Split it into groups of seven words
    space_separated_list = clean_brief_description.split()
    NUMBER_OF_WORDS = 7
    seven_word_groups = [
        ' '.join(space_separated_list[x:x+NUMBER_OF_WORDS]) for x in range(
            0,
            len(space_separated_list),
            NUMBER_OF_WORDS,
        )
    ]
    work_metadata_strings += seven_word_groups

# Join our metadata together with the '|' symbol for this Notebook
ACMI_METADATA = '|'.join(work_metadata_strings)

print(
    f'ACMI API input string from https://api.acmi.net.au/works/{work["id"]}/\n'
    f'{ACMI_METADATA}'
)

# Generate an image
print('Generating an image...')
subprocess.run(['python', 'generate.py', '-p', ACMI_METADATA], check=True)
