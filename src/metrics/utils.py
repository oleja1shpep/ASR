import editdistance

# Based on seminar materials

# Don't forget to support cases when target_text == ''


def calc_cer(target_text, predicted_text) -> float:
    assert target_text != "", "target text is empty, unable to calc cer"
    return editdistance.eval(target_text, predicted_text) / len(target_text)


def calc_wer(target_text, predicted_text) -> float:
    assert target_text != "", "target text is empty, unable to calc wer"
    return editdistance.eval(target_text.split(), predicted_text.split()) / len(
        target_text.split()
    )
