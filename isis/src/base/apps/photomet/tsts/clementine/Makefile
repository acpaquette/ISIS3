APPNAME = photomet

include $(ISISROOT)/make/isismake.tsts

commands:
	$(APPNAME) from=$(INPUT)/lub2303j.191.lev1.cub \
	to=$(OUTPUT)/lub2303j.191.lev1.phot.cub \
	frompvl=$(INPUT)/clem.pvl normname=moonalbedo \
	phtname=lunarlambertmcewen > /dev/null;
