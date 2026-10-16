# Afterglow — Song Recommender

A focused Next.js music-discovery interface for describing a vibe, starting from a seed song, recording recommendation feedback, and collecting tracks into private playlists.

## Run locally

1. Copy `.env.example` to `.env` and set `DATABASE_URL`, `DIRECT_URL`, and a random `JWT_SECRET` (32+ characters).
2. Install packages: `npm install`
3. Apply the Prisma migration and generate the client: `npm run prisma:migrate && npm run prisma:generate`
4. Start the application: `npm run dev`

Run checks with `npm test`, `npm run lint`, and `npm run build`.

## Environment

The app uses a server-only Prisma connection. For a Supabase serverless deployment, use the transaction-mode connection in `DATABASE_URL` (with `pgbouncer=true`) and a session/direct connection in `DIRECT_URL` for migrations. `RECOMMENDATION_API_URL` is optional; when absent, the UI shows explicitly marked development mock results so the discovery and playlist loop can be tested without claiming they are live recommendations.

Authentication is app-owned: passwords are hashed with bcrypt, sessions are signed JWTs in HTTP-only cookies, and every playlist API route verifies the signed-in user owns the target playlist.
