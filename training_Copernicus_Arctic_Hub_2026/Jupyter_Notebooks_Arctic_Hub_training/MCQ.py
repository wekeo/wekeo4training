questions = [
    {
        "question": "Which month is considered the standard indicator for the annual Arctic Sea Ice minimum?",
        "options": [
            "March",
            "June",
            "September",
            "December"
        ],
        "answer": 2,
        "explanation": "September marks the end of the summer melting season, revealing the remaining multi-year ice pack and serving as the primary indicator for long-term climate trends."
    },
    {
        "question": "What primary positive feedback mechanism accelerates Arctic sea ice loss?",
        "options": [
            "The ice-albedo feedback",
            "The salinity-density loop",
            "The geostrophic wind forcing",
            "The ozone depletion cycle"
        ],
        "answer": 0,
        "explanation": "As reflective ice melts, it reveals darker open ocean water, which absorbs more solar radiation, warming the surface and causing further ice melt."
    },
    {
        "question": "Which Copernicus product service provides 2-meter air temperature (T2m) atmospheric reanalyses?",
        "options": [
            "Copernicus Marine Service (CMEMS)",
            "Copernicus Climate Change Service (C3S / ERA5)",
            "Copernicus Land Monitoring Service (CLMS)",
            "Copernicus Emergency Management Service (CEMS)"
        ],
        "answer": 1,
        "explanation": "ERA5 is produced by the Copernicus Climate Change Service (C3S) at ECMWF, providing comprehensive global atmospheric reanalysis data."
    },
    {
        "question": "What is a main limitation of using high-degree polynomial regressions for future sea ice projections?",
        "options": [
            "They cannot be computed in Python.",
            "They smooth out seasonal cycles too effectively.",
            "They diverge dramatically and unpredictably outside the training time domain.",
            "They force the trend to always stay strictly linear."
        ],
        "answer": 2,
        "explanation": "Polynomial regressions are good interpolators within the dataset range, but fail catastrophically at extrapolation, often leading to unphysical results (like ice exploding toward infinity)."
    },
    {
        "question": "In a Hovmöller diagram plotting latitude vs. time for sea ice concentration, what does a northward shift of the 15% contour line indicate?",
        "options": [
            "An expansion of the ice pack toward lower latitudes",
            "A poleward retreat of the sea ice edge",
            "An increase in sea ice thickness",
            "A cooling trend in the sub-Arctic oceans"
        ],
        "answer": 1,
        "explanation": "The 15% sea ice concentration contour defines the sea ice extent margin. A northward (higher latitude) shift over time illustrates the physical retreat of the ice edge toward the North Pole."
    },
    {
        "question": "What strong statistical relationship is observed between summer (Jul-Aug) air temperature (T2m) and September Sea Ice Extent (SIE)?",
        "options": [
            "A strong positive correlation (R ≈ +0.88)",
            "No significant correlation (R ≈ 0.0)",
            "A strong negative correlation (R ≈ -0.88)",
            "A parabolic relationship with a minimum in 2000"
        ],
        "answer": 2,
        "explanation": "A strong negative Pearson correlation coefficient (R ≈ -0.88) demonstrates that warmer summer temperatures directly dictate lower sea ice minimums in September."
    }
]


# -----------------------------
# FONCTION PRINCIPALE
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
    # handler réponses
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
    # boutons réponses
    # -------------------------
    for i, opt in enumerate(q["options"]):
        btn = __import__("ipywidgets").Button(
            description=opt,
            layout=__import__("ipywidgets").Layout(width='auto')
        )
        btn.on_click(make_handler(i))
        buttons.append(btn)

    buttons_box.children = buttons