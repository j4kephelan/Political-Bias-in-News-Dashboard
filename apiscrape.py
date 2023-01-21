from pynytimes import NYTAPI
from datetime import date, datetime
import json
from bs4 import BeautifulSoup
import requests
import os

nyt = NYTAPI(
    key="bSBSdx57xh8Kg9Q8WGix9h35UF4K1Mex",  # Get your API Key at https://developer.nytimes.com
    parse_dates=True,
)

def nyt_api(topic, begin, end):
    """ use NYT api to get data for topic of interest

    :param topic: (str) topic of interest
            begin: (int) begin year of interest
            end: (int) end year of interest
    :return: return list of 50 articles data
    """

    # query api based off of parameters
    info = nyt.article_search(
        query=topic, results=100, dates={"start": date(begin, 1, 1), "end": date(end, 1, 1)}, options={
            "sources": [
                "New York Times"
            ],
        }
    )

    # populate list of all 50 articles
    info_lst = []
    for i in range(len(info)):

        # grab lead paragraph, pub_date, and length of lead paragraph
        paragraph = info[i]["lead_paragraph"]
        published = info[i]["pub_date"]
        leng = len(info[i]["lead_paragraph"])
        info_lst.append((paragraph, published, leng))

    # return data list
    return info_lst

def sort_length(tup):
    """
    sort list of data by length of paragraph shortest to longest
    :param tup: list of tuples
    :return: tup: sorted list of tuples
    """

    # lambda sorting by length of paragraph
    tup.sort(key=lambda x: x[2])
    return tup

def l_to_json(start, end, topic, lst):
    """
    takes list of tuples of data, year, and length, turns to json
    :param start: (int) start year
            end: (int) start year
            topic: (str) topic of interest
            lst: (list) list of tuples
    :return: none
    """

    # create dict for list of tuples
    dct = {}

    # count of correctly stored years
    count = 0

    # for each paragraph stored
    for i in lst:

        # if year is correct
        if i[1].year == start:

            # key is year - length of paragraph
            # val is paragraph
            dct[f"{i[1].year} - {i[2]}"] = i[0]

            # add one correctly stored val
            count += 1

            # break once 15 paragraphs are stored
            if count >= 15:
                break

    # serializing json
    json_object = json.dumps(dct, indent=4)

    # writing to sample.json
    with open(f"nyt_{topic}_{start}.json", "w") as outfile:
        outfile.write(json_object)

def l_get_data(topic, start, end):
    """
    get jsons of top 10 longest header paragraphs for a topic for each increment of 5 years from 1980-2023
    :param topic: (str) topic of interest
            start: (int) start year of interest
            end: (end) end year of interest
    :return: none
    """

    # start year
    y = start

    # for each year within desired range
    while y <= end:

        # update which json is currently being loaded in
        print("Current year loading:", y)

        # set start and end period
        front = y
        back = y + 1

        # get list of start paragraphs and article data
        lst = nyt_api(topic, front, back)

        # sort by length
        s_lst = sort_length(lst)

        # to json
        l_to_json(front, back, topic, s_lst)

        # add 1 year for each iteration
        y += 1

    print("Finished loading NYT data for", topic, "from", start,"to",end,"!")


def get_txt(year, s_page, topic):
    """
    get text data dictionary for first 8 articles of first page containing year of interest
    :param year: (str) year of interest
    :param s_page: (int) last searched page
    :param topic: (str) topic of interest
    :return: dct: (dct) dictionary of string data of 8 articles
            page: (int) last searched page number of website
    """

    # establish page to start search
    page = s_page

    # iterate until located page of articles with desired year
    while True:

        # create url
        url = f"https://nypost.com/search/{topic}/page/{page}/?orderby=date&order=asc&section=news"

        # get string data of webpage
        str_html = requests.get(url).text

        # use BS to get cleaner appearing html data
        soup = BeautifulSoup(str_html, features="html.parser")

        # divide by story
        divs = soup.find_all(class_="search-results__story")

        # divide into article title and publish date
        val1 = divs[10].find_all(class_="meta meta--byline")[0]

        # grab year published
        searched = str(val1).split("|")[0].split(" ")[-2]

        # output current search location
        print("page number:", page)
        print("publishing year:", searched)

        # if first article of page is for desired year, end loop
        if searched == year:
            print("located year!")
            break

        elif searched > year:
            print("no articles for selected year")
            break

        # index to next page
        page += 1

    if searched == year:
        # grab url of page that is for desired year
        url = f"https://nypost.com/search/{topic}/page/{page}/?orderby=date&order=asc&section=news"

        # get html string data for that year and create soup object
        str_html = requests.get(url).text
        soup = BeautifulSoup(str_html, features="html.parser")

        # create dct
        dct = {}

        # search for first 8 articles of page
        for idx in range(0,8):

            # locate story by index
            val1 = str(soup.find_all(class_="search-results__story")[idx]).split("href")[1].split(">\n\t\t\t\t\t")[0]

            # grab url of current story
            new_url = val1.split('="')[1][:-1]

            # account for formatting change of articles after 2012 (remove tab formatting)
            if int(year) > 2012:

                # clean new url
                new_url = new_url.split('"')[0]

            # request html data for current article and create soup object
            new_str_html = requests.get(new_url).text
            new_soup = BeautifulSoup(new_str_html, features="html.parser")

            # find location of text
            text = new_soup.find_all(class_="single__content entry-content m-bottom")

            # make string to add all text to
            long = ""

            # add all lines of cleaned text to long
            for i in range(len(str(text).split("<p>")[1:])):
                long += str(text).split("<p>")[1:][i].split("</p>")[0]

            # add to dictionary all text data with key being labeled as year - len(text)
            dct[f"{year} - {len(long)}"] = long
    else:
        dct = {}

    # return dict and current page searched
    return dct, page

def r_to_json(topic, s_page, year):
    """
    takes dct of data, turns to json
    :param topic: (str) topic of interest
            s_page: (int) last searched page
            year: (str) year of interest
    :return: page (int): last searhced page
    """

    # get dict and current searched page number for year of interest
    dct, page = get_txt(year, s_page, topic)

    # serializing json
    json_object = json.dumps(dct, indent=4)

    # writing to sample.json
    if "+" in topic:
        label = topic.replace("+", " ")
    else:
        label = topic

    with open(f"nyp_{label}_{year}.json", "w") as outfile:
        outfile.write(json_object)

    # return current searched page number
    return page

def r_get_data(topic, start, end):
    """
    get jsons data for defined range of years
    :param topic: (str) topic of interest
            start: (int) start year
            end: (int) end year
    :return: none
    """

    # start getting jsons for years 2002 and on
    y = start

    # start at page 1
    page = 1

    # iterate until year equals 2023
    while y <= end:

        # get jsons for each year
        page = r_to_json(topic, page, str(y))
        y +=1

    print("Finished loading NYP data for", topic, "from", start,"to", end,"!")

def main():


    #l_get_data("gay marriage", 2018, 2022)
    r_get_data("gay+marriage", 2002, 2005)


if __name__ == '__main__':
    main()

