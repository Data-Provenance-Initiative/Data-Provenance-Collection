import pandas as pd
import socket
import IP2Location
import csv
from tqdm import tqdm
import argparse

def get_country(ip_addresses):
    """
    Retrieves the country associated with each IP address in a list of domain-IP pairs.
    Uses the IP2Location database to determine the country.

    Parameters:
    ip_addresses (list of tuples): List containing tuples of domain and its corresponding IP address.

    Returns:
    list of tuples: List containing tuples of domain and corresponding country name.
    """
    # initialize IP2Location - default db (country level) saved locally at "data/IP2LOCATION-LITE-DB1.IPV6.BIN"
    ip2location = IP2Location.IP2Location()
    path_to_db = "data/IP2LOCATION-LITE-DB1.IPV6.BIN"
    ip2location.open(path_to_db)

    results = []
    for result in ip_addresses:
        domain = result[0]
        ip = result[1]
        rec = ip2location.get_all(ip)
        results.append((domain, rec.country_long))
    return results

def get_ip_from_urls(url_list):
    """
    Resolves the IP addresses for a list of URLs and tracks failures.

    Parameters:
    url_list (list of str): A list of URLs to resolve IP addresses for.

    Returns:
    tuple (list of tuples, list of str): Returns a tuple containing a list of tuples (each containing a URL and its resolved IP)
    and a list of URLs that failed to resolve.
    """
    ip_addresses = []
    failed_domains = []
    
    for url in tqdm(url_list):
        try:
            hostname = url.split("//")[-1].split("/")[0]
            ip_address = socket.gethostbyname(hostname)
            ip_addresses.append((url,ip_address))
        except:
            # print(f"failed to get ip address for domain: {url}")
            failed_domains.append(url)
            
    return ip_addresses, failed_domains

def get_default_url_list():
    """
    Reads a predefined CSV file to get a list of domain names.

    Returns:
    list of str: Returns a list of domain names extracted from the CSV file.
    """
    file_path = 'data/_top_2000_c4_token_and_urlcounts - top_2000_c4_token_and_urlcounts.csv'
    url_df = pd.read_csv(file_path, usecols=['Domain'])
    url_list = url_df['Domain'].tolist()
    return url_list

def read_urls_from_csv(file_path):
    """
    Reads URLs from a CSV file assuming no header and URLs in the first column.

    Parameters:
    file_path (str): Path to the CSV file.

    Returns:
    list of str: List of URLs read from the CSV file.
    """
    url_df = pd.read_csv(file_path, header=None)
    url_list = url_df.iloc[:, 0].tolist() # URLs must be in the first col with no header
    return url_list

def save_data(data, 
              name=None, 
              nationality_data=True):
    """
    Saves the provided data to a CSV file.

    Parameters:
    data (list of tuples): Data to save (each tuple contains domain and country).
    name (str, optional): Filename to save the data under. Defaults to 'domain_nationalities.csv' if not specified.
    nationality_data (bool, optional): If True, includes a header row with 'Domain' and 'Nationality'.

    Returns:
    None
    """
    if name:
        file_name = 'data/' + name
    else:
        file_name = 'data/domain_nationalities.csv'

    with open(file_name, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if nationality_data == True:
            writer.writerow(['Domain', 'Nationality'])
        for row in data:
            writer.writerow(row)
    print(f"Successfully saved data to: {file_name}")

def main(args):
    """
    Main function to orchestrate the fetching of country data for domains based on URLs or CSV file inputs.

    Parameters:
    args (Namespace): Command line arguments parsed by argparse.

    Returns:
    tuple (list of tuples, list of str): Tuple containing list of countries associated with domains and list of domains that failed IP resolution.
    """
    if args.url_csv:
        url_list = read_urls_from_csv(args.url_csv)
    elif args.custom_url_list:
        url_list = args.custom_url_list
    else:
        url_list = get_default_url_list()
    
    ip_addresses, failed_domains = get_ip_from_urls(url_list)
    print(f"Successfully converted {len(ip_addresses)}/{len(url_list)} domains to IP addresses")
    
    countries = get_country(ip_addresses)
    
    if args.write_file and args.output_filename:
        save_data(countries, name=args.output_filename)
    elif args.write_file:
        save_data(countries)
    
    return countries, failed_domains

if __name__ == "__main__":
    """ 
    Example commands:
       - Running the default URL list and writing the output to a file: python website_geolocation.py --write_file
       - Using a CSV file containing URLs: python website_geolocation.py --url_csv "path/to/your/url_file.csv"
       - Using a CSV file and writing the output to a file: python website_geolocation.py --url_csv "path/to/your/url_file.csv" --write_file
       - Specifying a custom list of URLs: python website_geolocation.py --custom_url_list "http://example.com" "http://example.org"
    """
    parser = argparse.ArgumentParser(description='Get country data for domains.')
    parser.add_argument('--custom_url_list', nargs='+', help='Custom URL list to process.')
    parser.add_argument('--url_csv', type=str, help='Path to a CSV file containing custom URLs to process. URLs must be in the first column with no header')
    parser.add_argument('--write_file', action='store_true', help='Whether to write the results to a CSV file')
    parser.add_argument('--output_filename', type=str, help='Filename for saving the output CSV file')
    args = parser.parse_args()
    main(args)