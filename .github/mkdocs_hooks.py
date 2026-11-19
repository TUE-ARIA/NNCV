"""MkDocs hook: point links to code files (not part of the site) to GitHub instead.

The Markdown pages link to files such as `../atlas/metrics.py`. That works when reading on
GitHub, but those files are not in the website, so here they are rewritten to GitHub URLs.
"""
import os
import posixpath
import re
from urllib.parse import quote

LINK = re.compile(r"(\]\()([^)\s]+)(\))")


def in_site(files, path):
    """True if `path` is a page or asset that is published (not excluded by `exclude_docs`)."""
    f = files.get_file_from_path(path)
    return f is not None and not f.inclusion.is_excluded()


def on_page_markdown(markdown, page, config, files):
    docs_dir = os.path.relpath(config["docs_dir"], os.path.dirname(config["config_file_path"])).replace(os.sep, "/")
    base = config["repo_url"].rstrip("/") + "/blob/main/" + quote(docs_dir)
    page_dir = posixpath.dirname(page.file.src_uri)

    def rewrite(match):
        target = match.group(2)
        if re.match(r"^[a-z]+:|^#|^/", target):  # absolute URLs, anchors, site-absolute paths
            return match.group(0)
        path, _, anchor = target.partition("#")
        resolved = posixpath.normpath(posixpath.join(page_dir, path))
        if in_site(files, resolved):
            return match.group(0)  # a page or image of the site
        if in_site(files, posixpath.join(resolved, "README.md")):
            return f"{match.group(1)}{posixpath.join(path, 'README.md')}{'#' + anchor if anchor else ''}{match.group(3)}"
        url = f"{base}/{quote(resolved)}" + (f"#{anchor}" if anchor else "")
        return f"{match.group(1)}{url}{match.group(3)}"

    return LINK.sub(rewrite, markdown)
