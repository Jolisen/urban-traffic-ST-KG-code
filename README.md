# Urban Traffic ST-KG Code

This repository contains the implementation code for constructing and applying the **Urban Traffic Spatio-Temporal Knowledge Graph (ST-KG)** in Neo4j, as described in our research. The code supports four main applications:  
1. **Spatiotemporal analysis of congestion dynamics**  
2. **Traffic speed prediction**  
3. **Intelligent question answering on congestion**  
4. **Tracing the causes of congestion**

---

## Main Requirements

The code has been tested with the following dependencies:
- Fiona==1.8.22
- folium==0.18.0
- GDAL==3.6.2
- geopandas==0.13.2
- shapely==2.0.3
- langchain==0.2.17
- openai==1.65.2
- py2neo==2021.2.4
- pyarrow==17.0.0
- pytorch-lightning==1.9.0
- torch==1.13.1+cu117
## Data
All datasets required to reproduce the results of the four applications in the paper are available at our Zenodo repository:
https://doi.org/10.5281/zenodo.16777726
## Program Structure & Usage

### **1. TRGK Creator**
Located in the `TRGK creater` folder.  
This code constructs the Urban Traffic ST-KG in [Neo4j](https://neo4j.com/):

- `road.ipynb` – Builds **road** entities and `touch` relationships between roads.  
- `grid.ipynb` – Builds **grid** entities and `adjacency` relationships between grids.  
- `POI.ipynb` – Builds **POI** entities and `contains` relationships between POIs and grids.  
- `state.ipynb` – Builds **state** entities, `next` relationships between states, and `located_in` relationships to grids.  
- `Cross relation construction.ipynb` – Builds `within` relationships between roads and grids.  
- `h-name in state.ipynb` – Embeds the ID of the grid entity with a `located_in` relationship into each state entity to optimize queries.  
- `TRKG index creater.ipynb` – Creates indexes for the constructed entities.
### **2. Congestion Level Assessment**

Located in `Congestion level assessment`:

- `v_acc.ipynb` – Calculates the average vehicle speed within each grid cell.

---










