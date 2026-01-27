import { apiEndpoint } from "./api-config";

export async function handleLogin(username :string, password: string) {
    try {
            const details = new URLSearchParams();
            details.append('username', username);
            details.append('password', password);

            const response = await fetch(apiEndpoint('auth/login'),{
                method: 'POST',
                headers: {
                    'accept': 'application/json',
                    'Content-Type': 'application/x-www-form-urlencoded',
                },
                body: details,
            });

            if (!response.ok) {
                // try to extract API error message, fall back to status
                let msg = `Login failed (${response.status})`;
                try {
                    const contentType = response.headers.get('content-type') || '';
                    if (contentType.includes('application/json')) {
                        const errorData = await response.json();
                        msg = errorData.message || errorData.detail || JSON.stringify(errorData) || msg;
                    } else {
                        const text = await response.text();
                        msg = text || msg;
                    }
                } catch (e) {}
                throw new Error(msg);
            }

            const data = await response.json();
            console.log('success: ', data);
            if (data && data.access_token) {
                return data.access_token;
            }
            // backend didn't return a token as expected
            throw new Error('Login succeeded but no access token returned by server');

        } catch (error: any) {
            console.error('error:', error?.message || error);
            throw error;
        }
}

export async function handleRegister(username: string, email: string, password: string) {
    try {
        const body = { username,password, email };

        const response = await fetch(apiEndpoint('auth/signup'), {
            method: 'POST',
            headers: {
                'accept': 'application/json',
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(body),
        });

        if (!response.ok) {
            let msg = `Registration failed (${response.status})`;
            try {
                const contentType = response.headers.get('content-type') || '';
                if (contentType.includes('application/json')) {
                    const errorData = await response.json();
                    msg = errorData.message || errorData.detail || JSON.stringify(errorData) || msg;
                } else {
                    const text = await response.text();
                    msg = text || msg;
                }
            } catch (e) {}
            throw new Error(msg);
        }

        const data = await response.json();
        console.log('registration success: ', data);
        if (data.access_token) return data.access_token;
        return data;

    } catch (error: any) {
        console.error('error:', error?.message || error);
        throw error;
    }
}