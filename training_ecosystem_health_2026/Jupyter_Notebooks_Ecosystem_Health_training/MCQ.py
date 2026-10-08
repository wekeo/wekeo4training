questions = [
    {
        "question": "Which WEkEO tool is used throughout this tutorial to search for and download Copernicus Land Monitoring Service products?",
        "options": [
            "The copernicusmarine toolbox",
            "The HDA (Harmonized Data Access) client",
            "The cartopy library",
        ],
        "answer": 1,
        "explanation": "The 'hda' Python client is used to load JSON search requests, query the WEkEO catalogue, and download the matching Urban Atlas and High Resolution Layer products."
    },
    {
        "question": "What kind of data is the Urban Atlas product, as used in Sections 3 and 4 of the notebook?",
        "options": [
            "A vector (polygon) land-use/land-cover database",
            "A raster time series of satellite reflectance",
            "A single global elevation model",
        ],
        "answer": 0,
        "explanation": "Urban Atlas is a detailed vector (GeoPackage) land-use/land-cover classification, loaded with GeoPandas and reclassified into simplified classes."
    },
    {
        "question": "In the notebook, into how many simplified classes is the detailed Urban Atlas nomenclature reclassified by the 'map_ua_code_to_class' function?",
        "options": [
            "2",
            "4",
            "8",
        ],
        "answer": 1,
        "explanation": "The function maps the detailed codes into 4 simplified classes: Urban/Artificial, Agricultural/Grassland, Forest/Semi-natural, and Wetlands/Water."
    },
    {
        "question": "Why must the 2012 and 2018 Urban Atlas vector layers be rasterized (with geocube) before building the land-cover transition matrix?",
        "options": [
            "Because pandas cannot read vector files",
            "To compare the two years pixel by pixel on a shared, identical grid",
            "Because rasters take up less disk space than vectors",
        ],
        "answer": 1,
        "explanation": "The transition matrix is built by comparing class codes at each grid cell between the two dates, which requires both years to share the exact same spatial grid — hence the rasterization step with geocube."
    },
    {
        "question": "In the land-cover transition matrix (Section 5), what do the OFF-diagonal cells represent?",
        "options": [
            "Pixels that stayed in the same class between 2012 and 2018",
            "Pixels that changed class between 2012 and 2018",
            "Pixels with missing (NoData) values",
        ],
        "answer": 1,
        "explanation": "The diagonal captures stability (same class in both years); the off-diagonal cells quantify the actual transitions, such as Agricultural to Urban or Forest to Urban."
    },
    {
        "question": "In Section 6 (landscape fragmentation), what combination of trends is described as the classic signature of fragmentation?",
        "options": [
            "More patches and a larger mean patch size",
            "Fewer patches and a larger mean patch size",
            "More patches and a smaller mean patch size",
        ],
        "answer": 2,
        "explanation": "An increasing number of separate patches together with a decreasing mean patch size shows that a class (e.g. forest) is being carved up into smaller, more isolated pieces."
    },
    {
        "question": "In the ecological connectivity graph (Section 7), what does a node represent?",
        "options": [
            "A single 100 m pixel of any land-cover class",
            "A forest patch of at least 50 ha ('reservoir')",
            "An urban administrative district",
        ],
        "answer": 1,
        "explanation": "Only sufficiently large forest patches (>= 50 ha) are kept as 'reservoir' nodes, used as a proxy for areas that can sustain a resident population of forest species."
    },
    {
        "question": "What does the friction (cost) surface introduced in Section 7.3 add to the analysis compared to a simple straight-line distance between reservoirs?",
        "options": [
            "It accounts for how hard each land-cover class is to cross (e.g. urban land is a major barrier)",
            "It measures the exact travel time of a car between two points",
            "It replaces the need for a connectivity graph entirely",
        ],
        "answer": 0,
        "explanation": "Each land-cover class is assigned a resistance value (e.g. Urban = 100, Forest = 1), so the corridor 'cost' reflects how difficult a species would find it to cross that landscape, not just the geographic distance."
    },
]


# -----------------------------
# MAIN FUNCTION
# -----------------------------
def load_question(state, feedback, question_html, buttons_box):

    q = questions[state["i"]]

    # reset feedback
    feedback.clear_output()

    # question text
    question_html.value = f"""
    <b>Question {state['i']+1}/{len(questions)}</b><br><br>
    {q['question']}
    """

    buttons = []

    # -------------------------
    # answer handler
    # -------------------------
    def make_handler(i):
        def handler(_):

            if state["answered"]:
                return

            state["answered"] = True

            feedback.clear_output()

            with feedback:

                if i == q["answer"]:
                    state["score"] += 1
                    print("✅ Correct!")
                else:
                    print("❌ Incorrect")

                print(q["explanation"])
                print(f"\nScore: {state['score']}/{len(questions)}")

                import ipywidgets as widgets

                next_btn = widgets.Button(description="Next ➡")

                def go_next(_):
                    state["i"] += 1
                    state["answered"] = False

                    if state["i"] < len(questions):
                        load_question(state, feedback, question_html, buttons_box)
                    else:
                        question_html.value = ""
                        buttons_box.children = []
                        feedback.clear_output()

                        with feedback:
                            print("🎓 Quiz completed!")
                            print(f"Final score: {state['score']}/{len(questions)}")

                next_btn.on_click(go_next)
                display(next_btn)

        return handler

    # -------------------------
    # answer buttons
    # -------------------------
    for i, opt in enumerate(q["options"]):
        btn = __import__("ipywidgets").Button(
            description=opt,
            layout=__import__("ipywidgets").Layout(width='auto')
        )
        btn.on_click(make_handler(i))
        buttons.append(btn)

    buttons_box.children = buttons