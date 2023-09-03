import plotly.graph_objects as go
import numpy as np
import time
import math


def create_slider_from_traces(traces):
    fig = go.Figure()
    for each in traces:
        fig.add_trace(each)

    # Create and add slider
    steps = []
    for index in range(len(fig.data)):
        step = dict(
            method="update",
            args=[
                {"visible": [False] * len(fig.data)},
                {"title": "Timestep: " + str(index)},
            ],  # layout attribute
        )
        step["args"][0]["visible"][index] = True  # Toggle i'th trace to "visible"
        steps.append(step)

    sliders = [
        dict(
            steps=steps,
        )
    ]

    fig.update_layout(sliders=sliders)
    return fig
    
scatters = []
# Create figure

# # Add traces, one for each slider step
# for step in range(0, 50, 1):
#     step = step / 10
#     scatters.append(
#         go.Scatter(
#             visible=False,
#             line=dict(color="#00CED1", width=6),
#             name="𝜈 = " + str(step),
#             x=list(each / 100 for each in range(0, 1000, 1)),
#             y=(list(math.sin(step * each / 100) for each in range(0, 1000, 1))),
#         )
#     )

   

#     fig.show()

#     time.sleep(10)
