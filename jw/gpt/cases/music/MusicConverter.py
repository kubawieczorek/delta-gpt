from mido import MidiFile, MidiTrack, MetaMessage, Message

# Define a list of MetaMessage types that can be removed without affecting sound
removable_meta_types = [
    'track_name', 'instrument_name', 'text', 'lyrics', 'marker',
    'cue_point', 'time_signature', 'key_signature', 'end_of_track'
]


class MusicConverter:

    def __init__(self,
                 time_reduction,
                 velocity_reduction,
                 control_change_val_reduction,
                 pitch_reduction,
                 note_reduction):
        self.dictionary = {}
        self.time_reduction = time_reduction
        self.velocity_reduction = velocity_reduction
        self.control_change_val_reduction = control_change_val_reduction
        self.pitch_reduction = pitch_reduction
        self.note_reduction = note_reduction

    # Function to extract all messages into a flat list with track info
    def extract_messages(self, midi_file_paths):
        music_keys = []

        for music_file_path in midi_file_paths:
            midi = MidiFile(music_file_path)
            music_keys.append('New song')
            # Store messages along with track index in a flat list
            for track_index, track in enumerate(midi.tracks):
                music_keys.append('New track')

                for msg in track:
                    msgs_simple = self.simplify_message(msg)
                    if msgs_simple is None:
                        continue

                    for msg_simple in msgs_simple:
                        key = str(msg_simple)
                        if key not in self.dictionary:
                            self.dictionary[key] = msg_simple

                        music_keys.append(key)

        return music_keys

    def simplify_message(self, msg):
        messages = []

        if hasattr(msg, 'time') and msg.time != 0:
            time = round(msg.time, self.time_reduction)
            msg.time = 0
            messages.append(f"wait_{time}")

        if hasattr(msg, 'channel'):
            channel = msg.channel
            msg.channel = 0
            messages.append(f"channel_{channel}")

        if msg.type == 'note_on':
            note = round(msg.note, self.note_reduction)
            velocity = round(msg.velocity, self.velocity_reduction)
            messages.append(f"note_on_{note}_{velocity}")

        elif msg.type == 'note_off':
            note = round(msg.note, self.note_reduction)
            messages.append(f"note_off_{note}")

        elif msg.type == 'control_change':
            control = msg.control
            value = round(msg.value, self.control_change_val_reduction)
            messages.append(f"controlchange_{value}_{control}")

        elif msg.type == 'pitchwheel':
            pitch = round(msg.pitch, self.pitch_reduction)
            messages.append(f"pitchwheel_{pitch}")
        # Simplify timing (quantized to musical note durations)
        # elif msg.type == 'time':
        #     # Map time to broad categories like quarter notes, eighth notes, etc.
        #     # Assuming 'time' here represents delta time in ticks; adjust as needed
        #     time = msg.time
        #     if time >= 480:
        #         return 'wait_quarter'
        #     elif time >= 240:
        #         return 'wait_eighth'
        #     elif time >= 120:
        #         return 'wait_sixteenth'
        #     else:
        #         return 'wait_tick'

        elif isinstance(msg, MetaMessage) and msg.type in removable_meta_types:
            return None
        else:
            messages.append(msg)
        return messages

    def create_midi_from_flat_messages(self, music_keys, output_file):
        new_midi = MidiFile()
        current_track = None

        # Organize messages back into their respective tracks
        current_wait = 0
        current_channel = 0

        for msg_key in music_keys:
            # if msg == 'New song':
            #     break
            if msg_key == 'New track' or current_track is None:
                # Create a new track if it doesn't exist
                current_track = MidiTrack()
                new_midi.tracks.append(current_track)
                continue

            msg = self.dictionary[msg_key]
            maybe_wait = self.desimplify_wait(msg)
            if maybe_wait is not None:
                current_wait += maybe_wait
                continue

            maybe_channel = self.desimplify_channel(msg)
            if maybe_channel is not None:
                current_channel = maybe_channel
                continue

            if isinstance(msg, str):
                desimplified_msg = self.desimplify_message(msg)
            else:
                desimplified_msg = msg

            desimplified_msg.time = current_wait
            if hasattr(msg, 'channel'):
                desimplified_msg.channel = current_channel

            current_track.append(desimplified_msg)
            current_wait = 0

        # Save the new MIDI file
        new_midi.save(output_file)

    def desimplify_wait(self, token):
        if not isinstance(token, str):
            return None

        parts = token.split('_')
        action = parts[0]
        value = parts[1]
        if action == 'wait':
            return int(value)

        return None

    def desimplify_channel(self, token):
        if not isinstance(token, str):
            return None

        parts = token.split('_')
        action = parts[0]
        value = parts[1]
        if action == 'channel':
            return int(value)

        return None

    def desimplify_message(self, token):
        parts = token.split('_')
        action = parts[0]

        if action == 'note':
            action_type = parts[1]
            note = int(parts[2])
            if action_type == 'on':
                velocity = int(parts[3])
                return Message('note_on', note=note, velocity=velocity)

            elif action_type == 'off':
                return Message('note_off', note=note, velocity=0)

        elif action == 'controlchange':
            value = int(parts[1])
            control = int(parts[2])
            return Message('control_change', value=value, control=control)

        elif action == 'pitchwheel':
            pitch = int(parts[1])
            return Message('pitchwheel', pitch=pitch)

        # # Reconstruct timing (assuming a specific tick value for simplicity)
        # elif action == 'wait':
        #     action_type = parts[1]
        #     wait_type = action_type
        #     time = {'quarter': 480, 'eighth': 240, 'sixteenth': 120}.get(wait_type, 480)
        #     return time  # Time value, which will be used in the track loop

        print('UNEXPECTED desimplify message')
        return token


def round(value, rounding):
    return int(value / rounding) * rounding
