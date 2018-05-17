from datetime import datetime
import re
import time
import urllib.request
from bs4 import BeautifulSoup

def get_cookies():
    """
    Get cookies from cookies.txt.
    """
    with open('cookies.txt', 'r') as file:
        session = file.readline().strip()
        remember_token = file.readline().strip()

    return (session, remember_token)

def get_page(session, remember_token):
    """
    Retrieve page with urllib with given cookies.
    """
    opener = urllib.request.build_opener()
    opener.addheaders.append(('Cookie', 'session={}'.format(session)))
    opener.addheaders.append(('Cookie', 'remember_token={}'.format(remember_token)))
    page = opener.open("https://contest.openai.com/user/job").read()

    return page

def page_to_info(page):
    """
    Use BeautifulSoup4 to parse info from HTML page.
    """
    soup = BeautifulSoup(page, 'html5lib')
    info = {}

    # Get summary info
    job_div = soup.find('div', {'class': 'job'})
    job_div_dds = job_div.find('dl').find_all('dd')

    status = job_div_dds[0].contents[0].strip()
    if status == 'finished':
        docker_name = job_div_dds[1].contents[0].strip()
        start_time = job_div_dds[2].contents[0].strip()
        eta = ''
        score = job_div_dds[3].contents[0].strip()
    else:
        docker_name = job_div_dds[1].contents[0].strip()
        start_time = job_div_dds[2].contents[0].strip()
        eta = job_div_dds[3].contents[0].strip()
        score = job_div_dds[4].contents[0].strip()

    info['summary'] = {}
    info['summary']['status'] = status
    info['summary']['docker_name'] = docker_name
    info['summary']['start_time'] = start_time
    info['summary']['eta'] = eta
    info['summary']['score'] = score

    # Get individual task infos
    worker_info_table = soup.find('table', {'class': 'worker_info'})
    worker_info_table_trs = worker_info_table.find_all('tr')

    info['tasks'] = []
    for i in range(1, len(worker_info_table_trs)):
        tds = worker_info_table_trs[i].find_all('td')
        task = tds[0].contents[0].strip()
        status = tds[1].contents[0].strip()
        score = tds[2].contents[0].strip()
        progress = tds[3].contents[0].strip()
        eta = tds[4].contents[0].strip().replace(' ', '').replace('\n', '')
        error = ''.join(tds[5].contents)

        task_info = {}
        task_info['task'] = task
        task_info['status'] = status
        task_info['score'] = score
        task_info['progress'] = progress
        task_info['eta'] = eta
        task_info['error'] = error
        info['tasks'].append(task_info)

    return info

def info_to_log(info):
    """
    Formats info to a single-line string to save.
    """
    log = '{:30} | {:20} | {:10} | {:>7}'.format(
        str(datetime.now()),
        info['summary']['docker_name'],
        info['summary']['status'],
        info['summary']['score'],
    )
    for i in range(5):
        log += ' | {:2} | {:10} | {:5} | {:>7}'.format(
            info['tasks'][i]['task'],
            info['tasks'][i]['status'],
            info['tasks'][i]['progress'],
            info['tasks'][i]['score'],
        )
    log += '\n'

    return log

def save_log(log):
    with open('info.txt', 'a+') as file:
        file.write(log)

def auto_log(interval_in_sec):
    """
    Log lastest info automatically with given interval in seconds.
    """
    while True:
        session, remember_token = get_cookies()
        page = get_page(session, remember_token)
        info = page_to_info(page)
        log = info_to_log(info)
        save_log(log)

        time.sleep(interval_in_sec)

if __name__ == '__main__':
    session, remember_token = get_cookies()
    page = get_page(session, remember_token)
    info = page_to_info(page)
    log = info_to_log(info)
    save_log(log)
