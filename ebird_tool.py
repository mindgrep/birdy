import requests
import os
from agno.tools import tool
from rich.pretty import pprint
import json


baseUrl = "https://api.ebird.org/v2"
apiToken = os.environ["EBIRD_API_TOKEN"]

# Load the taxonomy data
with open("ebird_json/taxonomy.json", "r") as f:
    taxonomy_data = json.load(f)


def make_request(endpoint: str) -> str:
    url = f"{baseUrl}/{endpoint}"
    resp = requests.get(
        url,
        headers={"X-eBirdApiToken": apiToken},
    )
    pprint(f"Query: {url}, response: {resp.json()}")
    return resp.json()


@tool(
    show_result=False,
    stop_after_tool_call=False,
    description="Get the species code for a given common name.",
)
def get_species_code(common_name: str) -> str:
    """Get the species code for a given common name."""
    for item in taxonomy_data:
        if item["comName"].lower() == common_name.lower():
            return item["SPECIES_CODE"]
    return None


@tool(
    show_result=False,
    stop_after_tool_call=False,
    description="Get recent sightings of a bird by species code from eBird",
)
def get_sightings(speciesCode: str, regionCode: str) -> str:
    print(f"get_sightings called with params {speciesCode} and {regionCode}")
    records = make_request(f"data/obs/{regionCode}/recent/{speciesCode}")
    # Optimize the output to reduce token usage
    optimized_records = [
        {
            "locId": r["locId"],
            "locName": r["locName"],
            "obsDt": r["obsDt"],
            "howMany": r.get("howMany", "N/A"),
        }
        for r in records
    ]
    return json.dumps(optimized_records)


@tool(
    show_result=False,
    stop_after_tool_call=False,
    description="Get all species codes for birds in eBird",
    cache_results=True,
    cache_dir="ebird_cache",
)
def get_species_codes() -> str:
    print("get_species_codes called")
    return make_request("ref/taxonomy/ebird?fmt=json")


@tool(
    show_result=False,
    stop_after_tool_call=False,
    description="Get all region codes within a US state",
    cache_results=True,
    cache_dir="ebird_cache",
)
def get_region_codes(stateCode: str) -> str:
    print(f"get_region_codes called for state {stateCode}")
    # return make_request(f"ref/region/list/subnational2/US-{stateCode}")
    return "[{'code': 'US-TX-201', 'name': 'Harris County'}]"
