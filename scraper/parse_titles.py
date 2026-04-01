
import re
import pandas as pd

INPUT_CSV = "./final_results.csv"
OUTPUT_CSV = "./final_results_parsed.csv"

HAS_HEADER = True
TITLE_COL = "title"

MULTIWORD_MAKES = [
    "LAND ROVER","MERCEDES-BENZ","MERCEDES-BENZ","ALFA ROMEO","ASTON MARTIN","ROLLS ROYCE","AM GENERAL",
]
SINGLE_MAKES = {
    "ACURA","AUDI","BMW","BUICK","CADILLAC","CHEVROLET","CHRYSLER","DODGE","FIAT","FORD",
    "GENESIS","GMC","HONDA","HUMMER","HYUNDAI","INFINITI","JAGUAR","JEEP","KIA","LEXUS",
    "LINCOLN","MAZDA","MERCURY","MINI","MITSUBISHI","NISSAN","OLDSMOBILE","PLYMOUTH",
    "PONTIAC","PORSCHE","RAM","SATURN","SCION","SMART","SUBARU","SUZUKI","TESLA","TOYOTA",
    "VOLKSWAGEN","VOLVO","SAAB","ISUZU","FREIGHTLINER","MASERATI","BENTLEY","FERRARI",
    "LAMBORGHINI","LOTUS","MCLAREN","MAYBACH","RIVIAN","POLARIS"
}
TRIM_STARTERS = {
    "BASE","L","LE","LS","LT","LT1","LT2","LTZ","LX","EX","EX-L","EXL","S","SE","SEL","SES",
    "SL","SLE","SLT","SV","SR","SR5","ST","STX","XLT","XL","XLE","XSE","LIMITED","PREMIUM",
    "PRESTIGE","PLATINUM","TOURING","DENALI","AT4","ELEVATION","TRAILHAWK","TRD","Z71","ZR2",
    "RST","RS","SS","SRT","GT","GTS","GTI","GL","GLS","HSE","HST","AUTOBIOGRAPHY","PURE",
    "DYNAMIC","X-DYNAMIC","LUX","LUXURY","SIGNATURE","OVERLAND","ALTITUDE","SUMMIT","RUBICON",
    "MOJAVE","SAHARA","WILLYS","BIG","HORN","LARAMIE","LONGHORN","TRADESMAN","REBEL",
    "AWD","FWD","RWD","4WD","4X4","2WD","QUATTRO","TFSI","TRONIC","CVT","HYBRID","PHEV","EV",
    "DIESEL","TDI","TSI","V6","V8","I4","I6","250","300","350","450","550","2500","3500","HD",
    "CLASSIC","CABRIOLET","COUPE","SEDAN"
}
MULTIWORD_MODELS = [
    "RANGE ROVER SPORT","RANGE ROVER EVOQUE","RANGE ROVER VELAR","RANGE ROVER",
    "GRAND CHEROKEE L","GRAND CHEROKEE","GRAND CARAVAN","TOWN & COUNTRY","SANTA FE","SANTA CRUZ",
    "OUTLANDER SPORT","WRANGLER UNLIMITED","PRO MASTER CITY","PRO MASTER","TRANSIT CONNECT",
    "ECONOLINE WAGON","SILVERADO 1500","SILVERADO 1500HD","SILVERADO 2500","SILVERADO 2500HD",
    "SILVERADO 3500","SILVERADO 3500HD","SIERRA 1500","SIERRA 1500HD","SIERRA 2500","SIERRA 2500HD",
    "SIERRA 3500","SIERRA 3500HD","YUKON XL","ESCALADE ESV","F 150","F 250","F 250 SUPER DUTY",
    "F 350","F 350 SUPER DUTY","F 450","F 450 SUPER DUTY","MUSTANG MACH-E","MODEL 3","MODEL S",
    "MODEL X","MODEL Y","CX 3","CX 5","CX 9","MX 5","QX 50","QX 60","QX 80","RX 350","RX 450H",
    "ES 350","IS 250","IS 350","GS 350","A CLASS","C CLASS","E CLASS","S CLASS",
]
YEAR_RE = re.compile(r"^(19|20)\d{2}\b")
SPACE_RE = re.compile(r"\s+")

def normalize(text: str) -> str:
    return SPACE_RE.sub(" ", str(text).strip().upper())

def extract_make(rest: str):
    for make in MULTIWORD_MAKES:
        if rest == make or rest.startswith(make + " "):
            return make, rest[len(make):].strip()
    first = rest.split(" ", 1)[0]
    return first, rest[len(first):].strip()

def split_model_and_rest(rest: str):
    if not rest:
        return "", ""

    for model in sorted(MULTIWORD_MODELS, key=lambda x: len(x.split()), reverse=True):
        if rest == model or rest.startswith(model + " "):
            return model, rest[len(model):].strip()

    tokens = rest.split()

    if len(tokens) >= 2 and re.fullmatch(r"[A-Z]+", tokens[0]) and re.fullmatch(r"[0-9]{1,4}[A-Z]*", tokens[1]):
        return " ".join(tokens[:2]), " ".join(tokens[2:])

    cut = None
    for i, tok in enumerate(tokens):
        if i == 0:
            continue
        if tok in TRIM_STARTERS or re.fullmatch(r"\d\.\d[T]?", tok) or re.fullmatch(r"V\d|I\d", tok):
            cut = i
            break

    if cut is None:
        return rest, ""
    return " ".join(tokens[:cut]), " ".join(tokens[cut:])

def parse_title(title: str):
    title = normalize(title)
    if not title:
        return pd.Series({"year": None, "make": None, "model": None, "rest": None})

    m = YEAR_RE.match(title)
    if not m:
        return pd.Series({"year": None, "make": None, "model": title, "rest": ""})

    year = int(m.group(0))
    after_year = title[4:].strip()

    make, after_make = extract_make(after_year)
    model, extra = split_model_and_rest(after_make)

    return pd.Series({"year": year, "make": make, "model": model, "rest": extra})

def main():
    if HAS_HEADER:
        df = pd.read_csv(INPUT_CSV)
        source_col = TITLE_COL
    else:
        df = pd.read_csv(INPUT_CSV, header=None, names=[TITLE_COL])
        source_col = TITLE_COL

    parsed = df[source_col].astype(str).apply(parse_title)
    out = pd.concat([df, parsed], axis=1)
    out.to_csv(OUTPUT_CSV, index=False)

    print(f"Completed: {OUTPUT_CSV}")
    print(out[[source_col, "year", "make", "model", "rest"]].head(20).to_string(index=False))

if __name__ == "__main__":
    main()
