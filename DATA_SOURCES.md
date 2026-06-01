# Data sources, attribution & licenses

The **pysephone source code is licensed under the MIT License** (see `LICENSE`).

This package also **bundles a number of third-party reference datasets** under
`src/pysephone/data/`. Those datasets are **not** covered by the MIT license —
each retains the terms of its original source, as documented below. If you use,
redistribute, or build upon pysephone, you are responsible for honoring the terms
of any bundled dataset you rely on, including any attribution and use
restrictions noted here.

> ⚠️ **Note on commercial use.** Most bundled datasets permit reuse with
> attribution, but **`liestal.csv` is licensed for non-commercial use only**, and
> **`kyoto.csv`** is an academic dataset provided for research use with required
> citations and **no explicit redistribution grant**. These two files are
> redistributed here on those terms; commercial users in particular must review
> them before relying on these files.

---

## GMU cherry blossom data — `src/pysephone/data/gmu_cherry/data/`

Cleaned peak-bloom / first-flowering records (schema:
`location, lat, long, alt, year, bloom_date, bloom_doy`). Per-file provenance:

| File | Source | License / terms | Attribution / citation |
|---|---|---|---|
| `washingtondc.csv` | US EPA, Climate Change Indicators — Cherry Blossoms | US federal source (public domain); see source for details | "Source: U.S. EPA, Climate Change Indicators in the United States — https://www.epa.gov/climate-indicators/cherry-blossoms" |
| `meteoswiss.csv` | MeteoSwiss / opendata.swiss (phenological observations) | **Open use, incl. commercial**; must provide source | "Source: MeteoSwiss" |
| `japan.csv` | Japan Meteorological Agency | Cite source | "Source: Japan Meteorological Agency — https://www.data.jma.go.jp/sakura/data/pdf/005.pdf" |
| `south_korea.csv` | Korea Meteorological Administration | Cite source | "Source: Korean Meteorological Administration" |
| `liestal.csv` | Landwirtschaftliches Zentrum Ebenrain, Sissach & MeteoSwiss | **Non-commercial use only**; must provide source | "Source: Landwirtschaftliches Zentrum Ebenrain, Sissach and MeteoSwiss" |
| `kyoto.csv` | Yasuyuki Aono, Osaka Prefecture University | Academic / research use; **no explicit redistribution grant**; cite the papers below | Aono & Saito (2010), *Int. J. Biometeorology* 54:211–219; Aono & Kazui (2008), *Int. J. Climatology* 28:905–914. Source: http://atmenv.envi.osakafu-u.ac.jp/aono/kyophenotemp4/ |

See `src/pysephone/data/gmu_cherry/data/README.md` for the full per-source notes.
This collection originates from George Mason University's public
`peak-bloom-prediction` competition repository.

---

## PEP725 metadata — `src/pysephone/data/pep725/metadata/`

Only **lookup/metadata tables** (PEP725 species codes, country codes, and the
catalog of entries to download) are bundled here. The actual PEP725 phenology
**observations are not redistributed** — they are downloaded at runtime using
your own PEP725 account credentials (see `src/pysephone/data/pep725/README.md`).
PEP725 data is subject to the PEP725 data policy (https://www.pep725.eu/).

---

## World administrative boundaries — `src/pysephone/data/resources/`

`world-administrative-boundaries.geojson` — obtained from OpenDataSoft
(https://public.opendatasoft.com/explore/dataset/world-administrative-boundaries/),
derived from Natural Earth (public domain) and related sources. Used only for
optional map backgrounds in visualizations. Please review OpenDataSoft's terms
for the dataset before redistribution.
