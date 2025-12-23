export async function handleLogin(username :string, password: string) {
    try {
            const details = new URLSearchParams();
            details.append('username', username);
            details.append('password', password);

            const response = await fetch('http://127.0.0.1:8000/v1/auth/login',{
                method: 'POST',
                headers: {
                    'accept': 'application/json',
                    'Content-Type': 'application/x-www-form-urlencoded',
                },
                body: details,
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.message || 'login failed');
            }

            const data = await response.json();
            console.log('success: ', data);
            if(data.access_token) {
                return data.access_token;
            }

        } catch (error: any) {
            console.error('error:', error.message);
        }
}