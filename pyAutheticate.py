import ee

ee.Initialize()
print(ee.Image('NASA/GPM_L3/IMERG_MONTHLY_V06').getInfo())

