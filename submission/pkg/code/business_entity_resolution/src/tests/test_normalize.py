import pytest

from entity_resolution.normalize import address_state, house_number, name_key, name_skeleton, normalize_address, normalize_name


@pytest.mark.parametrize(
    "variant, expected",
    [
        ("Svl Tubes Pvt Ltd.", "svl tubes"),
        ("Svl  Tubes Pvt", "svl tubes"),
        ("Tirth Resources Private-Limited", "tirth resources"),
        ("The Choice Transit-Enterprises", "choice transit enterprises"),
        ("Jex First Fócus", "jex first focus"),
        ("-- Holloway Peak Inc Seafood", "holloway peak seafood"),
        ("<< Team Ecole", "team ecole"),
        ("Thermal & Fils SASU", "thermal fils"),
        ("Tunisie Inter Cie International SARL", "tunisie inter international"),
        ("@jexfirst", "jexfirst"),
        ("bnpgroupcompaniesdelhi.com", "bnpgroupcompaniesdelhi"),
    ],
)
def test_normalize_name_strips_noise_legal_suffixes_and_accents(variant, expected):
    assert normalize_name(variant) == expected


def test_normalize_name_romanizes_indic_script_and_drops_romanized_legal_words():
    assert normalize_name("राम मार्केटिंग प्राइवेट लिमिटेड") == "ram marketimg"
    assert normalize_name("சிவ் டிரேடிங் பிரைவேட் லிமிடெட்") == "civ tiretin"


def test_name_skeleton_aligns_romanized_indic_with_latin_spelling():
    assert name_skeleton("લક્ષ્મી ડેવલપર્સ") == name_skeleton("Laxmi Developers")
    assert name_skeleton("सन बेस्ट") == name_skeleton("Sun Best")


@pytest.mark.parametrize(
    "a, b",
    [
        ("920 STINOCHER SAINT, SOLN, IA", "920 Stinocher St, Soln, Iowa"),
        ("177B, Aazad Path, Jaipur, RJ", "177B, AAZAD PATH, JAIPUR, Rajasthan"),
        ("PLOT NO: ##51 FIRST FLOOR, CHENNAI, TN", "Plot No: 51 First Floor, Chennai, Tamil Nadu"),
        ("63 R. DE DIEPPE, LILLE", "63 Rue de Dieppe, Lille"),
        ("12 AV LEON JOUHAUX, LILLE", "12 Avenue Léon Jouhaux, Lille"),
        ("2138 Talkington Dr, Flagstaff, AZ", "2138 Talkington Drive, Flagstaff, Arizona"),
    ],
)
def test_normalize_address_maps_abbreviations_states_and_punctuation_to_one_form(a, b):
    assert normalize_address(a) == normalize_address(b)


def test_normalize_address_drops_leading_zeros_of_house_numbers():
    assert normalize_address("002800 Wisconsin Ave, WASHINGTON, DC") == normalize_address("2800 Wisconsin Avenue, Washington, DC")
    assert normalize_address("039-12-10, VIJAYAWADA") == normalize_address("39-12-10, Vijayawada")


@pytest.mark.parametrize(
    "address, expected",
    [
        ("4896 Crater Avenue, Keizer, OR", 4896),
        ("Brownfield, TX, Unit 2010, 2010 6th Street", 2010),
        ("Unit 4, CO, Clifton, 547 Pearce Avenue", 547),
        ("2600 Napa Valley Way, Unit B-47, Columbia, TN", 2600),
        ("Napa Valley Way, Unit B-47, Columbia, Tennessee", None),
        ("2991 Hosreshoe Dr, PO Box 7616, Worcester Township, Pennsylvania", 2991),
        ("GRANADA CT, PEORIA, IL", None),
        ("##3008 Granada Court, NULL, Peoria, Illinois", 3008),
        ("N° 12 RUE DU CHAUFONR, LILLE, Hauts-de-France", 12),
        ("(520) RUE DE CASSEL, DUNKERQUE", 520),
        ("4008 1/2 HIGHWAY 230, AR, WALNUT RIDGE", 4008),
        ("", None),
    ],
)
def test_house_number_is_the_first_street_number_ignoring_unit_and_box_numbers(address, expected):
    assert house_number(normalize_address(address)) == expected


@pytest.mark.parametrize(
    "address, expected",
    [
        ("20 Rue Parmentier, Dunkerque, Hauts-de-France", "hdf"),
        ("Nord, DUNKERQUE, 164 PLAUE DES ALOUETTES", "hdf"),
        ("13 ALLÉE DES PINS, Bordeaux, Gironde", "naq"),
        ("15 ROUTE DE LA BOSSE, LA BAULE-ESCOUBLAC, Loire-Atlantique", "pdl"),
        ("22 RUE Descartes, Hauts-de-France, Calais", "hdf"),
        ("56 RUE JEAN CASTANDET CAZAUX, LA TESTE-DE-BUCH", ""),
    ],
)
def test_address_state_reads_french_regions_and_departements(address, expected):
    assert address_state(normalize_address(address), "France") == expected


@pytest.mark.parametrize(
    "a, b",
    [
        ("Urgent Care Health Inc", "URGENT CARE HEALTH-INC"),
        ("Up Chemicals Public Limited", "Limited Up Chemicals Public"),
        ("Kasturi Automation Private Limited", "Kasturi Automation Pvt. Ltd."),
        ("Transcontinental Staffing Union LLC", "llc transcontinental staffing union"),
    ],
)
def test_name_key_is_equal_for_reordered_and_abbreviated_copies(a, b):
    assert name_key(a) == name_key(b)


@pytest.mark.parametrize(
    "a, b",
    [
        ("Urgent Care Health Inc", "Urgent Care Health LLC"),
        ("Glyphelle Disciplined Inc", "Glyphelle Disciplined Northside"),
    ],
)
def test_name_key_keeps_legal_forms_and_extra_words_apart(a, b):
    assert name_key(a) != name_key(b)
