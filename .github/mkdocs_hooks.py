"""MkDocs hooks: links to code files go to GitHub, and section titles get the right index pages.

The Markdown pages link to files such as `../atlas/metrics.py`. That works when reading on
GitHub, but those files are not in the website, so here they are rewritten to GitHub URLs.
"""
import os
import posixpath
import re
from urllib.parse import quote, unquote

from mkdocs.structure.pages import Page

LINK = re.compile(r"(\]\()([^)\s]+)(\))")


def in_site(files, path):
    """True if `path` is a page or asset that is published (not excluded by `exclude_docs`)."""
    f = files.get_file_from_path(path)
    return f is not None and not f.inclusion.is_excluded()


def on_page_markdown(markdown, page, config, files):
    docs_dir = os.path.relpath(config["docs_dir"], os.path.dirname(config["config_file_path"])).replace(os.sep, "/")
    base = config["repo_url"].rstrip("/") + "/blob/main"
    if docs_dir != ".":
        base += "/" + quote(docs_dir)
    page_dir = posixpath.dirname(page.file.src_uri)

    def rewrite(match):
        target = match.group(2)
        if re.match(r"^[a-z]+:|^#|^/", target):  # absolute URLs, anchors, site-absolute paths
            return match.group(0)
        path, _, anchor = target.partition("#")
        resolved = posixpath.normpath(posixpath.join(page_dir, unquote(path)))  # links may encode spaces as %20
        if in_site(files, resolved):
            return match.group(0)  # a page or image of the site
        if in_site(files, posixpath.join(resolved, "README.md")):
            return f"{match.group(1)}{posixpath.join(path, 'README.md')}{'#' + anchor if anchor else ''}{match.group(3)}"
        url = f"{base}/{quote(resolved)}" + (f"#{anchor}" if anchor else "")
        return f"{match.group(1)}{url}{match.group(3)}"

    return LINK.sub(rewrite, markdown)


def on_nav(nav, config, files):
    """Keep a section title plain when its README is just one of several separate pages.

    With `navigation.indexes`, the theme puts the first README of a section behind the section title.
    That suits a folder with its overview (a topic and its subtopics), but not a list of folders such as
    the tracks, where every page is a README. There the READMEs stay ordinary entries in the list.
    """
    def visit(items):
        for item in items:
            if not item.is_section:
                continue
            pages = [c for c in item.children if c.is_page]
            readme = next((p for p in pages if p.is_index), None)
            if readme:
                folder = posixpath.dirname(readme.file.src_uri) + "/"
                if any(not p.file.src_uri.startswith(folder) for p in pages):   # separate folders: no index
                    for p in pages:
                        if p.is_index:
                            p.__class__ = PlainPage
            visit(item.children)

    visit(nav.items)
    return nav


class PlainPage(Page):
    """A README page that is not shown as the index of its section."""
    is_index = False
