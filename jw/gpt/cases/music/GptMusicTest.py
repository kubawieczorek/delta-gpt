import mido
from mido import MidiFile, MidiTrack, format_as_string, parse_string

from jw.gpt.cases.music.MusicConverter import MusicConverter

# Specify the input and output MIDI file paths
input_file = 'songs2/test_piano.midi'
output_file = 'output-1.mid'

music_converter = MusicConverter(time_reduction=1,
                                 velocity_reduction=20,
                                 control_change_val_reduction=20,
                                 pitch_reduction=500,
                                 note_reduction=3
                                 )
# Extract messages from the input MIDI file into a flat list
flat_messages = music_converter.extract_messages([input_file])

# Create a new MIDI file from the flat list of messages
music_converter.create_midi_from_flat_messages(flat_messages, output_file)

print(f"New MIDI file created with a flat list of messages preserving tracks: {output_file}")
