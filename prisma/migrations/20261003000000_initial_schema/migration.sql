CREATE TYPE "FeedbackAction" AS ENUM ('SAVE', 'REJECT', 'OPEN', 'SKIP');

CREATE TABLE "User" (
  "id" TEXT NOT NULL,
  "name" TEXT NOT NULL,
  "email" TEXT NOT NULL,
  "passwordHash" TEXT NOT NULL,
  "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  "updatedAt" TIMESTAMP(3) NOT NULL,
  CONSTRAINT "User_pkey" PRIMARY KEY ("id")
);
CREATE TABLE "Track" (
  "id" TEXT NOT NULL, "title" TEXT NOT NULL, "artist" TEXT NOT NULL, "album" TEXT,
  "coverArtUrl" TEXT, "durationSeconds" INTEGER, "previewUrl" TEXT, "providerId" TEXT,
  "providerUrl" TEXT, "source" TEXT, "license" TEXT,
  "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP, "updatedAt" TIMESTAMP(3) NOT NULL,
  CONSTRAINT "Track_pkey" PRIMARY KEY ("id")
);
CREATE TABLE "Playlist" (
  "id" TEXT NOT NULL, "ownerId" TEXT NOT NULL, "title" TEXT NOT NULL,
  "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP, "updatedAt" TIMESTAMP(3) NOT NULL,
  CONSTRAINT "Playlist_pkey" PRIMARY KEY ("id")
);
CREATE TABLE "PlaylistTrack" (
  "playlistId" TEXT NOT NULL, "trackId" TEXT NOT NULL, "position" INTEGER NOT NULL,
  "addedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  CONSTRAINT "PlaylistTrack_pkey" PRIMARY KEY ("playlistId", "trackId")
);
CREATE TABLE "FeedbackEvent" (
  "id" TEXT NOT NULL, "userId" TEXT NOT NULL, "trackId" TEXT NOT NULL, "queryId" TEXT,
  "action" "FeedbackAction" NOT NULL, "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  CONSTRAINT "FeedbackEvent_pkey" PRIMARY KEY ("id")
);
CREATE UNIQUE INDEX "User_email_key" ON "User"("email");
CREATE UNIQUE INDEX "Track_providerId_providerUrl_key" ON "Track"("providerId", "providerUrl");
CREATE INDEX "Track_title_artist_idx" ON "Track"("title", "artist");
CREATE INDEX "Playlist_ownerId_updatedAt_idx" ON "Playlist"("ownerId", "updatedAt");
CREATE UNIQUE INDEX "PlaylistTrack_playlistId_position_key" ON "PlaylistTrack"("playlistId", "position");
CREATE INDEX "PlaylistTrack_trackId_idx" ON "PlaylistTrack"("trackId");
CREATE INDEX "FeedbackEvent_userId_createdAt_idx" ON "FeedbackEvent"("userId", "createdAt");
CREATE INDEX "FeedbackEvent_trackId_action_idx" ON "FeedbackEvent"("trackId", "action");
ALTER TABLE "Playlist" ADD CONSTRAINT "Playlist_ownerId_fkey" FOREIGN KEY ("ownerId") REFERENCES "User"("id") ON DELETE CASCADE ON UPDATE CASCADE;
ALTER TABLE "PlaylistTrack" ADD CONSTRAINT "PlaylistTrack_playlistId_fkey" FOREIGN KEY ("playlistId") REFERENCES "Playlist"("id") ON DELETE CASCADE ON UPDATE CASCADE;
ALTER TABLE "PlaylistTrack" ADD CONSTRAINT "PlaylistTrack_trackId_fkey" FOREIGN KEY ("trackId") REFERENCES "Track"("id") ON DELETE CASCADE ON UPDATE CASCADE;
ALTER TABLE "FeedbackEvent" ADD CONSTRAINT "FeedbackEvent_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE CASCADE ON UPDATE CASCADE;
ALTER TABLE "FeedbackEvent" ADD CONSTRAINT "FeedbackEvent_trackId_fkey" FOREIGN KEY ("trackId") REFERENCES "Track"("id") ON DELETE CASCADE ON UPDATE CASCADE;
