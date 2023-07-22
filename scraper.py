import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin


def scrape_pdf_files(url, download_folder):
    # Create the download folder if it doesn't exist
    if not os.path.exists(download_folder):
        os.makedirs(download_folder)

    # Send a GET request to the URL
    response = requests.get(url)
    response.raise_for_status()

    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')
    # print(soup)
    # Find all anchor tags with href attributes
    for link in soup.find_all('a', href=True):
        # print(link)
        href = link['href']

        # Check if the link ends with .pdf extension
        if href.lower().endswith('.pdf'):
            # Join the absolute URL with the relative URL
            pdf_url = urljoin(url, href)

            # Download the PDF file
            download_file(pdf_url, download_folder)

        # Check if the link is a nested URL
        elif urljoin(url, href).startswith(url):
            print(urljoin(url, href))
            # Recursively scrape the nested URL
            scrape_pdf_files(urljoin(url, href), download_folder)


def download_file(url, download_folder):
    # Get the file name from the URL
    filename = url.split('/')[-1]

    # Download the file
    response = requests.get(url)
    response.raise_for_status()

    # Save the file to the download folder
    file_path = os.path.join(download_folder, filename)
    with open(file_path, 'wb') as file:
        file.write(response.content)

    print(f"Downloaded: {filename}")


# Main entry point of the script
if __name__ == '__main__':
    print("This script will scrape PDF files from a given URL.")
    # URL to scrape for PDF files
    top_level_url = "https://open.umn.edu/opentextbooks/subjects"

    # Folder to store the downloaded PDF files
    download_folder = "pdf_files"

    # Scrape the PDF files and download them
    scrape_pdf_files(top_level_url, download_folder)
