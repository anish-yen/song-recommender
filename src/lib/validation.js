import { z } from 'zod';

export const credentialsSchema = z.object({
  name: z.string().trim().min(2, 'Name must be at least 2 characters.').max(80).optional(),
  email: z.string().trim().email('Enter a valid email address.').max(254),
  password: z.string().min(8, 'Password must be at least 8 characters.').max(128),
});

export const normalizeEmail = (email) => email.trim().toLowerCase();

export function validateCredentials(input, needsName = false) {
  const result = credentialsSchema.safeParse(input);
  if (!result.success) return { error: result.error.issues[0].message };
  if (needsName && !result.data.name) return { error: 'Please enter your name.' };
  return { data: { ...result.data, email: normalizeEmail(result.data.email) } };
}
