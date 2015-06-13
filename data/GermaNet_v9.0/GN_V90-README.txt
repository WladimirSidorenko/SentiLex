GermaNet Version 9.0
====================

Release date: April 2014

GermaNet is an on-line lexical-semantic database for German that has been structured
along the same lines as the Princeton WordNet for English.

GermaNet models nouns, verbs, and adjectives in their basic semantic relations like
hyponymy, meronymy, and antonymy, and constitutes a basic resource for word sense
disambiguation within NLP applications.

The development of the GermaNet resource has been directed by Prof. Erhard Hinrichs and
has been supported by research grants awarded to Prof. Hinrichs by the Ministerium für
Wissenschaft und Kunst Baden-Württemberg (MWK), the Bundesministerium für Bildung und
Forschung (BMBF), the Deutsche Forschungsgemeinschaft (DFG), and the European Commission.
The following researchers and research assistants (listed in alphabetic order) have
contributed to the construction of GermaNet and accompanying software:

Reinhild Barkey, Agnia Barsukova, Valérie Béchet-Tsarnos, Anne Brock, Edo Collins,
Bettina Demmer, Valentin Deyringer, Sabrina Galasso, Piklu Gupta, Annabell Grasse,
Helmut Feldweg, Clemens Frey, Marina Haid, Birgit Hamp, Verena Henrich, Marie Hinrichs,
Michael Hipp, Christina Hoppermann, Lars Horber, Silke Kugler, Claudia Kunze, Anja Laske,
Lothar Lemnitzer, Andrea Lorenz, Karin Naumann, Katrin Petodnig, Niko Schenk, Susanne
Schüle, Sarah Schulz, Daniil Sorokin, Rosemary Stegmann, Daniela Stelle, Daniela Stier,
Steffen Tacke, Christine Thielen, Tatiana Vodolazova, Andreas Wagner, Johannes Wahle,
Zarah-Leonie Weiss, Martin Wolf and Holger Wunsch participated in the construction of
GermaNet.


Quantitative Data
-----------------

Synsets: 93246
Lexical units: 121810
Literals: 110738
1,31 lexical units per synset
Number of conceptual relations: 105912
Number of lexical relations: 4258 (synonymy excluded)
Number of split compounds: 54759
Number of Interlingual Index (ILI) records: 28661
Number of Wiktionary sense descriptions: 29475


Significant Changes Compared to Version 8.0
-------------------------------------------

* New synsets and lexical units for all word categories, and new relations have been
  added.

* All nominal compounds have been split into their constituents (modifier and head) that
  are labelled with linguistic information such as foreign words and named entities.
  For more details, please see http://www.sfs.uni-tuebingen.de/GermaNet/compounds.shtml

* The German part of the Interlingual Index (ILI), which was initially created in the
  context of the EuroWordNet project, has been extended. It represents a mapping between
  wordnets of different languages. For detailed information see
  http://www.sfs.uni-tuebingen.de/GermaNet/ili.shtml


Data Structure
--------------

GermaNet 9.0 is provided both in an XML format and as a PostgreSQL database dump.

The GermaNet XML data can be found in the folder "GN_V90_XML". It consists of the
following modules: 54 GermaNet object files, 1 GermaNet relations file, 1 file
containing the Interlingual Index, 3 files containing the GermaNet-Wiktionary
mapping (including the Wiktionary sense descriptions that now extend GermaNet’s
lexical units), and 4 DTDs describing the format of these files.

The GermaNet database dump can be found in the folder "GN_V90_DB" along with
a short tutorial describing how to set up the database.


Supporting Programs
-------------------

On our homepage (http://www.sfs.uni-tuebingen.de/GermaNet/tools.shtml), the following
tools are provided:

* A Java Application Programming Interface (API) that provides easy access to all
  information in GermaNet

* A Semantic Relatedness API to calculate semantic relatedness between any two words/
  readings in GermaNet

* GermaNet-Explorer is a software to visualize GermaNet that was developed at the
  University of Dortmund (chair of Prof. Dr. Angelika Storrer)

* GernEdiT is an editor that provides a graphical user interface to the database


Contact and Information
-----------------------

There is further information about GermaNet at: http://www.sfs.uni-tuebingen.de/GermaNet

Please address your questions and comments to: germanetinfo@sfs.uni-tuebingen.de
