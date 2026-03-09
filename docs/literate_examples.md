This project will adopt literate examples. This means that the code or other configuration which would be shown to users
as examples will actually be considered test code.

The markdown formats used in the docs should have a simple way to offset fenced code sections which a user might use a
copypasta, and which should be verifiable in unit tests in this project.

The code will not be duplicated to a test. It will not be in multiple places. Instead, Each such example will be run in
a suitable context that verifies the example as valid.
Q