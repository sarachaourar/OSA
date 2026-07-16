#Imports
import pystac_client
import geogif
import stackstac
import os
import argparse

#Authentication
os.environ['GDAL_HTTP_TCP_KEEPALIVE'] = "YES"
os.environ['AWS_S3_ENDPOINT'] = "eodata.dataspace.copernicus.eu"
os.environ['AWS_ACCESS_KEY_ID'] = "your_access_key" # !
os.environ['AWS_SECRET_ACCESS_KEY'] = "your_secret_access_key" # !
os.environ['AWS_HTTPS'] = "YES"
os.environ['AWS_VIRTUAL_HOSTING'] = "FALSE"
os.environ['GDAL_HTTP_UNSAFESSL'] = "YES"

URL = "https://stac.dataspace.copernicus.eu/v1"
cat = pystac_client.Client.open(URL)
cat.add_conforms_to("ITEM_SEARCH")

#ask Igor about implimentation in the simulation interface
geom = {
    "type": "Polygon",
    "coordinates": [
        [
    [56.28128317814617, 26.28572205901952],
    [57.10073923060477, 26.28572205901952],
    [57.10073923060477, 25.793422786713066],
    [56.28128317814617, 25.793422786713066],
    [56.28128317814617, 26.28572205901952],
        ]
    ],
}


StartDay = input('Enter start date (yyyy-mm-dd) :')
StartTime = input('Enter start and time (hh:mm:ss) :')

#Sentinel-2 specific parameters
params_s2 = {
 "max_items": 100,
 "collections": "sentinel-2-l2a",
 "datetime": "2026-07-06/2026-07-10",
 "intersects": geom,
 "filter": {
            "op": "<",
            "args": [{"property": "eo:cloud_cover"}, 15] #try to implement a customisable function for cloud cover
            },
 "fields": {
 "include": [
                "id",
                "type",
                "geometry",
                "bbox",
                "properties.datetime",
                "properties.eo:cloud_cover",
                "assets.B02_20m",
                "assets.B03_20m",
                "assets.B04_20m"
            ]
            }
}

#Sentinel-1 specific parameters
params_s1 = {
 "max_items": 100,

 "collections": "sentinel-1-grd",

 "datetime": f"{StartDay}T{StartTime}""/2026-07-10",

 "intersects": geom,

 "fields": {
            "include": [
                        "id",
                        "type",
                        "geometry",
                        "bbox",
                        "properties.datetime",
                        "vv"
                        ]
            }
}

s2items = list(cat.search(**params_s2).items_as_dicts())
s1items = list(cat.search(**params_s1).items_as_dicts())

if len(s2items)<= 0 :
    print("The search returned 0 sentinel-2 images. ")
    print ("You can try changing the boundary box or the time frame.")
else : 
    oldest_s2item = min(s2items, key=lambda x: x["properties"]["datetime"])
    print(f"The search returned {len(s2items)} sentinel-2 images.") 

if len(s1items)<= 0 :
    print("The search returned 0 sentinel-2 images. ")
    print ("You can try changing the boundary box or the time frame.")
else : 
    oldest_s1item = min(s1items, key=lambda x: x["properties"]["datetime"])
    print(f"The search returned {len(s1items)} sentinel-1 images.") 

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.rl_config import defaultPageSize
from reportlab.lib.units import inch

PAGE_HEIGHT=defaultPageSize[1]; PAGE_WIDTH=defaultPageSize[0]
styles = getSampleStyleSheet()

Title = "Oil Spill Accidents Analyser"
pageinfo = "OSAA report"
def myFirstPage(canvas, doc):
    canvas.saveState()
    canvas.setFont('Times-Bold',16)
    canvas.drawCentredString(PAGE_WIDTH/2.0, PAGE_HEIGHT-108, Title)
    canvas.setFont('Times-Roman',12)
    canvas.drawString(inch, 0.75 * inch, "Page %d | %s" % (doc.page, pageinfo))
    canvas.restoreState()

def myLaterPages(canvas, doc):
    canvas.saveState()
    canvas.setFont('Times-Roman',12)
    canvas.drawString(inch, 0.75 * inch, "Page %d | %s" % (doc.page, pageinfo))
    canvas.restoreState()


def go():
    doc = SimpleDocTemplate(f"{oldest_s1item["id"][:4]}{oldest_s1item["properties"]["datetime"][:10]}_{oldest_s2item["id"][:4]}{oldest_s2item["properties"]["datetime"][:10]}.pdf")
    S1day = oldest_s1item["properties"]["datetime"][:10]
    S1time = oldest_s1item["properties"]["datetime"][11:19]
    S2day = oldest_s2item["properties"]["datetime"][:10]
    S2time = oldest_s2item["properties"]["datetime"][11:19]    
    Story = [Spacer(1,2*inch)]
    style = styles["Normal"]
    text = f'The sentinel-1 image was aquired on {S1day} at {S1time}. The sentinel-2 image was aquired on {S2day} at {S2time}'
    p = Paragraph(text, style)
    Story.append(p)
    Story.append(Spacer(1,0.2*inch))
    doc.build(Story, onFirstPage=myFirstPage, onLaterPages=myLaterPages)

if __name__ == '__main__':
    go()