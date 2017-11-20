import datetime
import logging
import random
from typing import Dict, List, Callable, Union
import os
import requests
import time
from requests import Response
from requests.auth import HTTPBasicAuth
from src import config


def decode(b: bytes) -> str:
    try:
        contents = b.decode("utf-8")
    except UnicodeDecodeError:
        logging.warning("Exception occurred while decoding, ignore the current file")
        contents = ""
    return contents


ext_lang_dict: Dict[str, Union[str, List[str]]] = {
    "rb": "Ruby",
    "py": "Python",
    "c": "C",
    "cpp": "C++",
    "java": "Java",
    "scala": "Scala",
    "kt": "Kotlin",
    "js": "Javascript",
    "sh": "Shell",
    "php": "PHP",
    "css": "CSS",
    "cs": "C#",
    "html": "HTML",
    "htm": "HTML",
    "xml": "XML",
    "yaws": "Erlang",
    "pl": "Perl",
    "ts": "Typescript",
    "go": "Go",
    "swift": "Swift",
    "r": "R",
    "m": ["Objective-C", "Matlab"],
    "vimrc": "VimL",
    "vim": "VimL",
    "coffee": "CoffeeScript",
    "tex": "Tex",
    "el": "Lisp",
    "lisp": "Lisp",
    "cl": "Lisp",
    "hs": "Haskell",
    "lhs": "Haskell",
    "lua": "Lua",
    "clj": "Closure",
    "cljs": "Closure",
    "cljc": "Closure",
    "edn": "Closure",
    "mat": "MatLab",
    "pde": "Arduino",
    "groovy": "Groovy",
    "rs": "Rust",
    "rlib": "Rust",
    "pp": "Puppet",
}


class GithubFetcher:
    def __init__(self, download_location: str, first_repo_id: int, repo_count: int, proxies: Dict[str, str],
                 username: str,
                 password: str):
        self.__download_location: str = download_location
        self.__first_repo_id: int = first_repo_id
        self.__repo_count: int = repo_count
        self.__proxies = proxies
        self.__username = username
        self.__password = password
        self.__filter: Dict[str, Callable] = {
            "php": extract_php
        }

    def run(self) -> None:
        repos: List[Dict[str, str]] = self.__get_repo_list()
        for repo in repos:
            logging.info(f"Start fetching repo {repo['id']}")
            type_to_files_list: Dict[str, Dict[str, str]] = self.__get_files(repo["contents_url"])
            for file_type, download_url_list in type_to_files_list.items():
                dir: str = os.path.join(self.__download_location, file_type)
                for download_url in download_url_list:
                    file_name = self.__add_time_to_filename(os.path.basename(download_url))
                    location = os.path.join(dir, file_name)
                    self.__download(download_url, location)
        logging.info("Download is completed.")

    def __add_time_to_filename(self, file_name: str) -> str:
        file_name_parts = os.path.splitext(file_name)
        time_batch = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        return f"{file_name_parts[0]}_{time_batch}{file_name_parts[1]}"

    def __get_repo_list(self) -> List[Dict[str, str]]:
        result: List[Dict[str, str]] = []
        start_id: int = self.__first_repo_id - 1 if self.__first_repo_id - 1 >= 0 else 0
        while True:
            repos = self.__do_get(f"https://api.github.com/repositories?since={start_id}").json()

            # [:-7] is to remove {+path} at the end of each repo
            result += [{"id": repo["id"], "contents_url": repo["contents_url"][:-7]} for repo in repos]
            if len(result) >= self.__repo_count:
                return result[:self.__repo_count]
            start_id = result[-1]["id"]

    def __get_files(self, contents_url: str, type_to_files_dict: Dict[str, List[str]] = None) -> Dict[str, List[str]]:
        logging.info(f"Start fetching files from contents_url: {contents_url}")
        if type_to_files_dict is None:
            type_to_files_dict = {}
        files: List[Dict] = self.__do_get(contents_url).json()

        if type(files) != list:
            # may be we exceeded the rate limit, sleep for 10 seconds and try again
            logging.info(f"Got response: {files}, will sleep for 10 seconds and try other repos")
            time.sleep(10)
            return {}

        for file in files:
            if file["type"] == "dir":
                self.__get_files(file["url"], type_to_files_dict)
            else:
                file_type = self.__get_file_type(file["name"])
                if file_type is None:
                    continue
                elif type(file_type) == list:
                    # TODO support multiple file types
                    file_type = file_type[0]
                self.__update_type_files_dict(type_to_files_dict, file_type, file["download_url"])
        return type_to_files_dict

    def __update_type_files_dict(self, type_to_files_dict: Dict[str, List[str]], file_type: str,
                                 download_url: str) -> None:
        if file_type in type_to_files_dict:
            type_to_files_dict[file_type].append(download_url)
        else:
            type_to_files_dict[file_type] = [download_url]

    def __get_file_type(self, filename: str) -> Union[str, List[str]]:
        ext = os.path.splitext(filename)[1][1:]
        return ext_lang_dict.get(ext)

    def __get_filter(self, filename: str) -> Callable:
        ext = os.path.splitext(filename)[1][1:]
        return self.__filter.get(ext)

    def __download(self, download_url: str, file_location: str) -> None:
        logging.info(f"Download file from {download_url}, saving to {file_location}")
        contents: bytes = self.__do_get(download_url).content

        # perform filtering
        contents_filter: Callable = self.__get_filter(file_location)
        if contents_filter is not None:
            contents = contents_filter(contents)
        contents = self.__universal_contents_filter(contents)

        os.makedirs(os.path.dirname(file_location), exist_ok=True)
        if len(contents) > 0:
            with open(file_location, "wb") as f:
                f.write(contents)

    def __do_get(self, url) -> Response:
        return requests.get(url, proxies=self.__proxies, auth=HTTPBasicAuth(self.__username, self.__password))

    def __universal_contents_filter(self, contents: bytes) -> bytes:
        # set line separator as \n instead of \r or \r\n
        return "\n".join(decode(contents).splitlines()).encode("utf-8")


def extract_between(s: str, str1: str, str2: str) -> List[str]:
    result: List[str] = []
    start_pos = 0
    while True:
        index1: int = s.find(str1, start_pos)
        index2 = -1 if index1 == -1 else s.find(str2, index1 + 1)
        if index1 == -1 or index2 == -1:
            break
        result.append(s[index1 + len(str1): index2])
        start_pos = index2 + len(str2)
    return result


def extract_php(contents_in_bytes: bytes) -> bytes:
    php_start_tag = "<?php"
    php_end_tag = "?>"
    s = decode(contents_in_bytes)
    php_code = "\n".join(extract_between(s, php_start_tag, php_end_tag))
    if len(php_code) == 0:
        # maybe this php file only has a start tag
        php_start_pos = s.find(php_start_tag)
        if php_start_pos != -1:
            php_code = s[php_start_pos + len(php_start_tag):]
    return php_code.encode("utf-8")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    while True:
        repo_id = random.randint(0, 98019238)
        github_fetcher = GithubFetcher(
            config.download_location,
            repo_id,
            2,
            config.proxies,
            config.github_username,
            config.github_password)
        try:
            github_fetcher.run()
        except Exception as e:
            logging.error(f"Encountered exception: {e}, will sleep for 10 seconds and retry")
            time.sleep(10)
