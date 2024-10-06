from fractions import Fraction

from music21 import converter, note, chord, instrument, meter, key, clef, tempo, stream


def mid_to_notes(mid_songs):
    notes = []
    part_num = 1
    for i, file in enumerate(mid_songs):
        try:
            midi = converter.parse(file)
            parts = instrument.partitionByInstrument(midi)
            if parts:  # file has instrument parts
                for part in parts.parts:  # Iterate through each part
                    notes_to_parse = part.recurse()
                    print(f'Processing INPUT PART: {part_num}, Notes: {len(notes_to_parse.notes)}')
                    part_num = part_num + 1
                    parse_notes(notes_to_parse, notes)
            else:  # file has notes in a flat structure
                notes_to_parse = midi.flat.notes
                parse_notes(notes_to_parse, notes)

        except Exception as e:
            print(f'FAILED: {i + 1}: {file}. Error: {e}')

    return notes


def parse_notes(notes_to_parse, notes):
    for element in notes_to_parse:
        if isinstance(element, note.Note):  # Handling notes
            notes.append(str(element.pitch))
        elif isinstance(element, chord.Chord):  # Handling chords
            notes.append('.'.join(str(n) for n in element.normalOrder))
        elif isinstance(element, note.Rest):  # Handling rests
            notes.append(f'rest:{element.quarterLength}')  # Append rest with duration
        elif isinstance(element, instrument.Instrument):  # Handling instruments
            notes.append(f"Instrument: {element.instrumentName}")  # Append instrument name
        elif isinstance(element, meter.TimeSignature):  # Handling time signatures
            notes.append(f"Time Signature: {element.ratioString}")  # Append time signature
        elif isinstance(element, key.KeySignature):  # Handling key signatures
            notes.append(f"Key Signature: {element.sharps} sharps")  # Append key signature
        elif isinstance(element, clef.Clef):  # Handling clefs
            notes.append(f"Clef: {element.name}")  # Append clef name
        elif isinstance(element, tempo.MetronomeMark):  # Handling MetronomeMark (tempo)
            notes.append(f"Tempo: {element.number} BPM")  # Append metronome (BPM)


def create_midi(prediction_output, output_file='output.mid'):
    output_notes = []
    current_instrument = instrument.Piano()  # Default instrument
    current_part = stream.Part()  # Create a part stream for the current instrument
    part_num = 1
    offset = 0

    for pattern in prediction_output:
        new_note = None  # Initialize new_note
        new_chord = None  # Initialize new_chord
        new_rest = None  # Initialize new_rest

        # Check for instrument changes
        if 'Instrument:' in pattern:
            instrument_name = pattern.split(': ')[1]
            current_instrument = getattr(instrument, instrument_name, instrument.Piano)()
            if current_part.notes:  # Append the previous part if it has notes
                output_notes.append(current_part)
                print(f'Appending OUTPUT PART {part_num}, Notes: {len(current_part.notes)}')
                part_num = part_num + 1
            current_part = stream.Part()  # Start a new part for the new instrument

        # Check if the pattern is a rest with duration
        elif pattern.lower().startswith('rest:'):
            duration_str = pattern.split(':')[1]
            try:
                rest_duration = float(Fraction(duration_str))  # Convert to float using Fraction
                new_rest = note.Rest()
                new_rest.offset = offset
                new_rest.quarterLength = rest_duration  # Set duration
                current_part.append(new_rest)
            except ValueError:
                print(f"Skipping unrecognized rest duration '{duration_str}'")

        # pattern is a chord
        elif ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = current_instrument
                new_note.offset = offset
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            current_part.append(new_chord)

        # pattern is a time signature
        elif 'Time Signature:' in pattern:
            time_sig = pattern.split(': ')[1]
            new_time_signature = meter.TimeSignature(time_sig)
            current_part.append(new_time_signature)

        # pattern is a key signature
        elif 'Key Signature:' in pattern:
            sharps = int(pattern.split(' ')[2])
            new_key_signature = key.KeySignature(sharps)
            current_part.append(new_key_signature)

        # pattern is a clef change
        elif 'Clef:' in pattern:
            clef_name = pattern.split(': ')[1]
            if clef_name.lower() == 'treble':
                new_clef = clef.TrebleClef()
            elif clef_name.lower() == 'bass':
                new_clef = clef.BassClef()
            else:
                new_clef = clef.TrebleClef()  # Default to TrebleClef if unrecognized
            current_part.append(new_clef)

        # pattern is a tempo (metronome mark)
        elif 'Tempo:' in pattern:
            bpm = int(pattern.split(' ')[1])
            new_tempo = tempo.MetronomeMark(number=bpm)
            current_part.append(new_tempo)

        # pattern is a note
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = current_instrument
            current_part.append(new_note)

        # Increase offset for the next note/rest/chord based on its duration
        if new_note is not None:
            offset += new_note.quarterLength  # Use the note's duration for offset increment
        elif new_chord is not None:
            offset += new_chord.quarterLength  # Use the chord's duration for offset increment
        elif new_rest is not None:
            offset += new_rest.quarterLength  # Adjust offset for rests as well

    # Append the last part to output_notes if it has notes
    if current_part.notes:
        output_notes.append(current_part)
        print(f'Appending OUTPUT PART {part_num}, Notes: {len(current_part.notes)}')

    midi_stream = stream.Score(output_notes)  # Create a Score to hold all parts
    midi_stream.write('midi', fp=output_file)
    print(f'MIDI file saved as {output_file}')
