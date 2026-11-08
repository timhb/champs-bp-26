"""Local NLM MeSH vocabulary lookup; case/report text never leaves the viewer."""

import re
from collections import defaultdict

WORDS = re.compile(r"[^\W_]+(?:['’][^\W_]+)*|&", re.UNICODE)
STOP_WORDS = set(
    "a an and are as at be been being by can case for from had has have in is it may no not of on or other our result results site source that the their there these this those to value was were which with without yes".split()
)
SHORT_TERMS = {"tb", "csf", "hiv", "cmv", "cns", "pcr", "ihc", "eye", "ear", "sex", "he"}


def tokens(text):
    return [match.group().casefold() for match in WORDS.finditer(text)]


def mesh_records(stream):
    """Stream descriptor XML into concept definitions and their official entry terms."""
    from xml.etree.ElementTree import iterparse

    for _, element in iterparse(stream, events=("end",)):
        if element.tag != "DescriptorRecord":
            continue
        descriptor = element.findtext("DescriptorUI")
        descriptor_name = element.findtext("DescriptorName/String", "")
        tree_numbers = [t.text for t in element.findall("TreeNumberList/TreeNumber") if t.text]
        concepts = element.findall("ConceptList/Concept")
        preferred = next((c for c in concepts if c.get("PreferredConceptYN") == "Y"), None)
        broader_definition = preferred.findtext("ScopeNote", "") if preferred is not None else ""
        for concept in element.findall("ConceptList/Concept"):
            uid = concept.findtext("ConceptUI")
            name = concept.findtext("ConceptName/String", "")
            if not uid or not name or not descriptor:
                continue
            yield {
                "id": uid,
                "descriptor": descriptor,
                "descriptor_name": descriptor_name,
                "tree_numbers": tree_numbers,
                "name": name,
                "definition": concept.findtext("ScopeNote", ""),
                "broader_definition": broader_definition
                if concept.findtext("ScopeNote") is None
                else "",
                "terms": sorted(
                    {name} | {t.text for t in concept.findall("TermList/Term/String") if t.text}
                ),
            }
        element.clear()


class MedicalGlossary:
    def __init__(self, data):
        self.metadata = {k: v for k, v in data.items() if k != "records"}
        self.records = {row["id"]: row for row in data["records"]}
        aliases = defaultdict(set)
        for row in self.records.values():
            # Fail closed for old caches and unclassified supplemental terms.
            # MeSH C covers diseases/pathological conditions; F03 covers mental disorders.
            if not any(
                tree.startswith("C") or tree == "F03" or tree.startswith("F03.")
                for tree in row.get("tree_numbers", [])
            ):
                continue
            for term in row["terms"]:
                words = tokens(term)
                if not words or len(words) > 30:
                    continue
                if len(words) == 1 and (
                    words[0] in STOP_WORDS or (len(words[0]) < 3 and words[0] not in SHORT_TERMS)
                ):
                    continue
                if len(words) == 1 and len(words[0]) < 3 and not term.isupper():
                    continue
                aliases[tuple(words)].add(row["id"])
        self.trie = {}
        for words, ids in aliases.items():
            node = self.trie
            for word in words:
                node = node.setdefault(word, {})
            node["_ids"] = sorted(ids)
        self.alias_count = len(aliases)
        self.search_terms = {
            row["id"]: [" ".join(tokens(term)) for term in row["terms"]]
            for row in self.records.values()
        }

    def entry(self, uid):
        row = self.records[uid]
        return row | {
            "url": row.get("url") or "https://meshb.nlm.nih.gov/record/ui?ui=" + row["descriptor"]
        }

    def matches(self, text):
        words = list(WORDS.finditer(text))
        matches, position = [], 0
        while position < len(words):
            node, best = self.trie, None
            for end in range(position, min(position + 30, len(words))):
                if end > position and not re.fullmatch(
                    r"[\s.\-–]*", text[words[end - 1].end() : words[end].start()]
                ):
                    break
                node = node.get(words[end].group().casefold())
                if node is None:
                    break
                if "_ids" in node:
                    value = text[words[position].start() : words[end].end()]
                    if (
                        end == position
                        and len(value) < 4
                        and value.casefold() not in {"eye", "ear", "sex"}
                        and not value.isupper()
                    ):
                        continue
                    best = (
                        {
                            "start": words[position].start(),
                            "end": words[end].end(),
                            "ids": node["_ids"],
                        },
                        end + 1,
                    )
            if best:
                match, position = best
                match["url"] = (
                    self.entry(match["ids"][0])["url"]
                    if len(match["ids"]) == 1
                    else "https://www.ncbi.nlm.nih.gov/mesh/"
                )
                matches.append(match)
            else:
                position += 1
        return matches

    def search(self, query, limit=20):
        query = " ".join(tokens(query))
        if not query:
            return []
        found = []
        for uid, terms in self.search_terms.items():
            name = " ".join(tokens(self.records[uid]["name"]))
            if query == name:
                rank = 0
            elif query in terms:
                rank = 1
            elif name.startswith(query):
                rank = 2
            elif any(all(" " + word in " " + term for word in query.split()) for term in terms):
                rank = 3
            else:
                continue
            found.append((rank, name, uid))
        return [self.entry(uid) for _, _, uid in sorted(found)[:limit]]
