# shazomi

Shazam for Omi: detect song names and timestamps based on snippets you hear throughout your day.

Very simple idea: hear constantly through the microphone and if the feed is above a certain threshold, it is not simply noise and we can fingerprint the audio snippet after sampling a sufficient chunk (based on Shazam's fingerprinting and detection algorithm abracadabra's requirements).

We can finally just store a timestamped (by time of day) list of these snippets with the song, artist, album and timestamp where the recorded snippet appears.
