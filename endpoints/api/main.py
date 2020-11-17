from fastapi import FastAPI

from .routes import datasets, experiments, node, models

app = FastAPI(
    title='Collective Learning API',
    description="""
The common collective learning API is used for both monitoring and controlling of models, datasets and experiments  
    """,
    version='0.1.0'
)


@app.get("/")
def index():
    """
    Simple endpoint, useful for API health checking.
    """
    return {'state': 'alive and kicking!'}


app.include_router(datasets.router)
app.include_router(experiments.router)
app.include_router(node.router)
app.include_router(models.router)
