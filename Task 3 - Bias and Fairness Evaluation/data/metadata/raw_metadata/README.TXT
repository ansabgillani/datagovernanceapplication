This file describes some of the more important bits of information in the
SQLite databases, stored in this directory. The plain text metadata files
(e.g. SPEAKERS.TXT) contain a subset of the information in these databases.

ia.sqlite.db
============

This file contains the metadata extracted, about the LibriVox recordings,
hosted on the Internet Archive. The database contains two tables:

* meta - each row represents a LV project stored on the Internet Archive.
         That is the URL of the main IA page for the project, the title and 
         the author of the book on which the recordings are based

* mp3 - each row is an audio chapter. It references the project this chapter
        is part of(i.e. a row in 'meta'), the URL from which the .mp3 file can
        be downloaded, it's size and checksums. For the 64kbit/s .mp3 recordings
        the "parent_id" points to another row, containg information about the
        128 kbit/s version from which it was derived.

pg.sqlite.db
============

Contains the parts of the metadata provided by Project Gutenbergs XML/RDF files,
which is relevant to the LibriSpeech corpus. The tables are:

* books - each row represents a PG book. Has columns for book's title, various
          classification codes, and the URLs for the ASCII and/or UTF-8 version
          of the text

* authors - each row represents an author, and have many-to-many relationship
            with the "books" table - i.e. a book can have more than one author
            and one author can be associated with more than one book.


lv-annotated.sqlite.db
======================

This database is a hodge-podge collection of various things that were relevant
for the alignment process and subsequent corpus creation.

First there are tables with information about LibriVox projects. This was
the original information in the database- the other things were then added
as needed:

* projects - each row describes a LibriVox audio book project. Has columns for
             the project's title, the URL for the associated LibriVox page, the
             number of the audio chapters, total time in seconds, the URL for
             relevant Internet Archive page and so on.

* audio_chapters - each row corresponds to an audio chapter a LibriVox project.
                   The "project_id" column contains the foreign key pointing to
                   the parent project, and there are columns containing the
                   duration of the chapter in seconds, the URL from which it
                   can be downloaded and so on. Another foreign key links the
                   chapter to a row in the "readers" table below.

* readers - contains the ID and the name of a LibriVox volunteer.

* authors - contains the name and dates of birth and death of a book author

* genres - ID and name of the genres under which the LibriVox projects are 
           classified


The tables that follow were used in the process of scheduling alignment jobs:

* jobs - contains the basic information about each scheduled job. Things like
         dates when the task was scheduled, enqueued on SGE, started and 
         finished, as well as status code for the outcome of the job(e.g.
         success or failure). The 'successor_id' field points the re-scheduled
         instance of this job if we had to restart it for some reason.

Two types of jobs were used to align the LibriSpeech's audio- in the parlance of
the source code they are called "top-half" and "bottom-half" jobs. The former
are responsible for normalizing a book's, building the phase 1 decoding graph,
and g2p lexicon generation(for more details search for the yet-to-be published
paper). The "bottom-half" jobs are user oriented, and are responsible for the
actual alignment. This design was chosen in order to share the initial processing
of a book, for which there could be LibriVox audio chapters, read by different
readers. The tables below extend the definition of "jobs", through the magic
of SQLAlchemy:

* top_half_jobs - contains the location of the source text to process.

* bottom_half_jobs - contains the ID of the reader, whose audio should be
                     processed by this job. Also stores number of statistics
                     about the job, filled after its completion. Information
                     like, the percentage of the audio successfully aligned
                     the number of the audio chapters for this reader which
                     succeeded and failed, the real-time factor, that is the
                     ratio between the useful, aligned audio and the time
                     it took to obtain it and so on. The "worst_verification_wer"
                     field records the result of the post-processing verification
                     of the obtained utterances(a process that produces a certain
                     number of false alarms)

* bottom_half_chapters - each row contains information about the processing
                         of an individual chapter within the bottom-half job.
                         Mostly the same information as above but at per-chapter
                         granularity.


The next set of tables contain information that was added using a process of
quickly reviewing tiny fraction of the audio to make sure that there aren't
multi-speaker recordings and other undesirable artifacts in the corpus. Note
that this process should not be considered infallible.


* reader_annotations - contains annotations about the individual reader.
                       Perhaps the only (more or less) reliable field here
                       is the gender information.

* audio_chapter_annotations - per-chapter "noisy" and "multi-speaker" flags.
