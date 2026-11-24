questions = [
    {
        "question": "Which WEkEO tool is used throughout this tutorial to search for and download products available via WEkEO?",
        "options": [
            "The copernicusmarine toolbox",
            "The HDA (Harmonized Data Access) client",
            "The cartopy library",
        ],
        "answer": 1,
        "explanation": "The 'hda' Python client is used to load JSON search requests, query the WEkEO catalogue, and download the matching products."
    },
    {
        "question": "In what format is the NDVI product provided?",
        "options": [
            "GPKG",
            "JSON",
            "NetCDF",
        ],
        "answer": 2,
        "explanation": "The NDVI product is provided in a NetCDF format."
    },
    {
        "question": "What function was used to plot several plots on the same figure?",
        "options": [
            "ax.pcolormesh()",
            "plt.plot()",
            "plt.subplots()",
        ],
        "answer": 2,
        "explanation": "plt.subplots() was used to create a figure and a set of subplots in a single call."
    },
    {
        "question": "What is the unit and typical value range of NDVI?",
        "options": [
            "No unit, from 0 to 1",
            "PSU, Unbounded",
            "No unit, unbounded",
        ],
        "answer": 0,
        "explanation": "NDVI stands for the Normalised Difference Vegetation Index, hence, it has no unit, and it is defined by a range [0, 1]"
    },
    {
        "question": "What xarray method allows us to subset data, in our case geographically?",
        "options": [
            ".sel()",
            ".isel()",
            ".subset()",
        ],
        "answer": 0,
        "explanation": "The .sel() method in Python's Xarray library selects data by label names along specified dimensions."
    },
    {
        "question": "What does seasonal reference cycle represent?",
        "options": [
            "The yearly mean value",
            "The difference between a year's mean and the climatology",
            "The average for each calendar month across all available years",
        ],
        "answer": 2,
        "explanation": "Seasonal reference cycle - the average NDVI for each calendar month across all available years."
    },
    {
        "question": "In this tutorial, what did positive NDVI anomaly mean?",
        "options": [
            "Vegetation is less green than the usual value for that calendar month",
            "Vegetation is greener than the usual value for that calendar month",
            "Vegetation greenness is close to the reference value",
        ],
        "answer": 1,
        "explanation": "Positive anomaly - vegetation is greener than the usual value for that calendar month."
    },
    {
        "question": "What type of correlation coefficient was used in Section 7?",
        "options": [
            "Pearson correlation",
            "Jakobson correlation",
            "Symmetric correlation",
        ],
        "answer": 0,
        "explanation": "The Pearson correlation coefficient is a statistical measure that quantifies the strength and direction of a linear relationship between two continuous variables."
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