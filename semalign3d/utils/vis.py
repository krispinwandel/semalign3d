def get_color(i: int, n: int):
    # return plotly color pallette depending on index
    return f"hsl({i*360//(n+5)}, 50%, 50%)"
