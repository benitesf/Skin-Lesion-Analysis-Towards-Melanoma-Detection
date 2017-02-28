# Script Name		: verbose.py
# Author				: Benites Fernandez, Edson
# Created				: 27/02/17
# Last Modified	: 
# Version				: 1.0

# Modifications	: 1.1 - some modifications
#							  : 1.2 - some modifications

# Description		: Implement function to set up the verbose									
#

import argparse 
import logging

def setUpVerbose():
	parser = argparse.ArgumentParser()
	parser.add_argument(
		'-d', '--debug',
		help = "Print lots of debugging statements",
		action = "store_const", dest = "loglevel", const = logging.DEBUG,
		default = logging.WARNING,
		)
	parser.add_argument(
		'-v', '--verbose',
		help = "Be verbose",
		action = "store_const", dest = "loglevel", const = logging.INFO,
		)
	args = parser.parse_args()
	logging.basicConfig(level = args.loglevel)
	return logging