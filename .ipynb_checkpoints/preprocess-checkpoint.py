{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "efee6a67-23fe-42d4-9d3e-bdf946831570",
   "metadata": {},
   "outputs": [],
   "source": [
    "import sqlite3"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "d2bcb17d-b3ee-4d27-9464-670ec36f1b09",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: faiss-cpu in c:\\users\\dell\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (1.10.0)\n",
      "Requirement already satisfied: numpy<3.0,>=1.25.0 in c:\\users\\dell\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from faiss-cpu) (2.2.4)\n",
      "Requirement already satisfied: packaging in c:\\users\\dell\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from faiss-cpu) (23.1)\n"
     ]
    }
   ],
   "source": [
    "!pip install faiss-cpu"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "6b40bdba-8d26-4897-b6da-ddd43319d1d9",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Collecting fastapi\n",
      "  Downloading fastapi-0.115.11-py3-none-any.whl.metadata (27 kB)\n",
      "Collecting uvicorn\n",
      "  Downloading uvicorn-0.34.0-py3-none-any.whl.metadata (6.5 kB)\n",
      "Requirement already satisfied: pandas in c:\\users\\dell\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (2.1.0)\n",
      "Collecting numpy==1.23.5\n",
      "  Downloading numpy-1.23.5-cp311-cp311-win_amd64.whl.metadata (2.3 kB)\n",
      "Requirement already satisfied: faiss-cpu in c:\\users\\dell\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (1.10.0)\n",
      "Collecting sentence-transformers\n",
      "  Downloading sentence_transformers-3.4.1-py3-none-any.whl.metadata (10 kB)\n",
      "Note: you may need to restart the kernel to use updated packages.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "ERROR: Could not find a version that satisfies the requirement sqlite3 (from versions: none)\n",
      "ERROR: No matching distribution found for sqlite3\n"
     ]
    }
   ],
   "source": [
    "pip install --no-cache-dir fastapi uvicorn pandas numpy==1.23.5 faiss-cpu sentence-transformers sqlite3 matplotlib seaborn"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "e84aa7d9-8a8c-4a2f-894e-eca4bc9513eb",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: pip in c:\\users\\dell\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (24.1.2)Note: you may need to restart the kernel to use updated packages.\n",
      "\n",
      "Collecting pip\n",
      "  Downloading pip-25.0.1-py3-none-any.whl.metadata (3.7 kB)\n",
      "Downloading pip-25.0.1-py3-none-any.whl (1.8 MB)\n",
      "   ---------------------------------------- 0.0/1.8 MB ? eta -:--:--\n",
      "   -------- ------------------------------- 0.4/1.8 MB 11.6 MB/s eta 0:00:01\n",
      "   ------------------------- -------------- 1.2/1.8 MB 15.1 MB/s eta 0:00:01\n",
      "   ---------------------------------------- 1.8/1.8 MB 16.8 MB/s eta 0:00:00\n",
      "Installing collected packages: pip\n",
      "  Attempting uninstall: pip\n",
      "    Found existing installation: pip 24.1.2\n",
      "    Uninstalling pip-24.1.2:\n",
      "      Successfully uninstalled pip-24.1.2\n",
      "Successfully installed pip-25.0.1\n"
     ]
    }
   ],
   "source": [
    "pip install --upgrade pip"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "2b6f15db-2594-47d0-a24a-e5cbaad6eaf7",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: numpy==1.23.5 in c:\\users\\dell\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (1.23.5)\n",
      "Requirement already satisfied: pandas==1.5.3 in c:\\users\\dell\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (1.5.3)\n",
      "Requirement already satisfied: python-dateutil>=2.8.1 in c:\\users\\dell\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from pandas==1.5.3) (2.8.2)\n",
      "Requirement already satisfied: pytz>=2020.1 in c:\\users\\dell\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from pandas==1.5.3) (2023.3.post1)\n",
      "Requirement already satisfied: six>=1.5 in c:\\users\\dell\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from python-dateutil>=2.8.1->pandas==1.5.3) (1.16.0)\n",
      "Note: you may need to restart the kernel to use updated packages.\n"
     ]
    }
   ],
   "source": [
    "pip install --no-cache-dir numpy==1.23.5 pandas==1.5.3"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "9dcded66-0c2d-4331-90d5-5d4bcd48cb78",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Collecting sentence-transformers\n",
      "  Downloading sentence_transformers-3.4.1-py3-none-any.whl.metadata (10 kB)\n",
      "Requirement already satisfied: transformers<5.0.0,>=4.41.0 in c:\\users\\dell\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from sentence-transformers) (4.41.0)\n",
      "Requirement already satisfied: tqdm in c:\\users\\dell\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from sentence-transformers) (4.66.1)\n",
      "Requirement already satisfied: torch>=1.11.0 in c:\\users\\dell\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from sentence-transformers) (2.0.1)\n",
      "Requirement already satisfied: scikit-learn in c:\\users\\dell\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from sentence-transformers) (1.3.0)\n",
      "Requirement already satisfied: scipy in c:\\users\\dell\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from sentence-transformers) (1.11.2)\n",
      "Requirement already satisfied: huggingface-hub>=0.20.0 in c:\\users\\dell\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from sentence-transformers) (0.23.0)\n",
      "Requirement already satisfied: Pillow in c:\\users\\dell\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from sentence-transformers) (10.0.0)\n",
      "Requirement already satisfied: filelock in c:\\users\\dell\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from huggingface-hub>=0.20.0->sentence-transformers) (3.12.4)\n",
      "Requirement already satisfied: fsspec>=2023.5.0 in c:\\users\\dell\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from huggingface-hub>=0.20.0->sentence-transformers) (2024.5.0)\n",
      "Requirement already satisfied: packaging>=20.9 in c:\\users\\dell\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from huggingface-hub>=0.20.0->sentence-transformers) (23.1)\n",
      "Requirement already satisfied: pyyaml>=5.1 in c:\\users\\dell\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from huggingface-hub>=0.20.0->sentence-transformers) (6.0.1)\n",
      "Requirement already satisfied: requests in c:\\users\\dell\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from huggingface-hub>=0.20.0->sentence-transformers) (2.31.0)\n",
      "Requirement already satisfied: typing-extensions>=3.7.4.3 in c:\\users\\dell\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from huggingface-hub>=0.20.0->sentence-transformers) (4.5.0)\n",
      "Requirement already satisfied: sympy in c:\\users\\dell\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from torch>=1.11.0->sentence-transformers) (1.12)\n",
      "Requirement already satisfied: networkx in c:\\users\\dell\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from torch>=1.11.0->sentence-transformers) (3.1)\n",
      "Requirement already satisfied: jinja2 in c:\\users\\dell\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from torch>=1.11.0->sentence-transformers) (3.1.2)\n",
      "Requirement already satisfied: colorama in c:\\users\\dell\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from tqdm->sentence-transformers) (0.4.6)\n",
      "Requirement already satisfied: numpy>=1.17 in c:\\users\\dell\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from transformers<5.0.0,>=4.41.0->sentence-transformers) (1.23.5)\n",
      "Requirement already satisfied: regex!=2019.12.17 in c:\\users\\dell\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from transformers<5.0.0,>=4.41.0->sentence-transformers) (2024.5.15)\n",
      "Requirement already satisfied: tokenizers<0.20,>=0.19 in c:\\users\\dell\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from transformers<5.0.0,>=4.41.0->sentence-transformers) (0.19.1)\n",
      "Requirement already satisfied: safetensors>=0.4.1 in c:\\users\\dell\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from transformers<5.0.0,>=4.41.0->sentence-transformers) (0.4.3)\n",
      "Requirement already satisfied: joblib>=1.1.1 in c:\\users\\dell\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from scikit-learn->sentence-transformers) (1.3.2)\n",
      "Requirement already satisfied: threadpoolctl>=2.0.0 in c:\\users\\dell\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from scikit-learn->sentence-transformers) (3.2.0)\n",
      "Requirement already satisfied: MarkupSafe>=2.0 in c:\\users\\dell\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from jinja2->torch>=1.11.0->sentence-transformers) (2.1.3)\n",
      "Requirement already satisfied: charset-normalizer<4,>=2 in c:\\users\\dell\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from requests->huggingface-hub>=0.20.0->sentence-transformers) (3.2.0)\n",
      "Requirement already satisfied: idna<4,>=2.5 in c:\\users\\dell\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from requests->huggingface-hub>=0.20.0->sentence-transformers) (3.4)\n",
      "Requirement already satisfied: urllib3<3,>=1.21.1 in c:\\users\\dell\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from requests->huggingface-hub>=0.20.0->sentence-transformers) (1.26.16)\n",
      "Requirement already satisfied: certifi>=2017.4.17 in c:\\users\\dell\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from requests->huggingface-hub>=0.20.0->sentence-transformers) (2023.7.22)\n",
      "Requirement already satisfied: mpmath>=0.19 in c:\\users\\dell\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from sympy->torch>=1.11.0->sentence-transformers) (1.3.0)\n",
      "Downloading sentence_transformers-3.4.1-py3-none-any.whl (275 kB)\n",
      "Installing collected packages: sentence-transformers\n",
      "Successfully installed sentence-transformers-3.4.1\n",
      "Note: you may need to restart the kernel to use updated packages.\n"
     ]
    }
   ],
   "source": [
    "pip install sentence-transformers"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "c58eb822-aef8-4b6c-b112-ca7c254dc989",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "✅ 'data' folder is ready!\n"
     ]
    }
   ],
   "source": [
    "import os\n",
    "\n",
    "# Create the 'data' directory if it doesn't exist\n",
    "if not os.path.exists(\"data\"):\n",
    "    os.makedirs(\"data\")\n",
    "\n",
    "print(\"✅ 'data' folder is ready!\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "a597ae34-a6b2-44f6-a6df-7b7e4403b4ec",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "✅ Processing 5 batches...\n",
      "   🔄 Processing batch 1/5...\n",
      "   🔄 Processing batch 2/5...\n",
      "   🔄 Processing batch 3/5...\n",
      "   🔄 Processing batch 4/5...\n",
      "⚠️ Skipping empty batch 5\n",
      "🎉✅ Preprocessing complete! Data cleaned, embeddings generated, and FAISS index saved.\n"
     ]
    }
   ],
   "source": [
    "# Process embeddings in batches\n",
    "batch_size = 250\n",
    "num_batches = len(df) // batch_size + 1\n",
    "embeddings_list = []\n",
    "\n",
    "print(f\"✅ Processing {num_batches} batches...\")\n",
    "\n",
    "for i in range(num_batches):\n",
    "    batch_texts = df['query_text'][i * batch_size:(i + 1) * batch_size].tolist()\n",
    "    \n",
    "    # Skip empty batches\n",
    "    if len(batch_texts) == 0:\n",
    "        print(f\"⚠️ Skipping empty batch {i+1}\")\n",
    "        continue\n",
    "\n",
    "    print(f\"   🔄 Processing batch {i+1}/{num_batches}...\")\n",
    "\n",
    "    batch_embeddings = model.encode(batch_texts, convert_to_numpy=True)\n",
    "\n",
    "    if batch_embeddings.size == 0:  # Check if embeddings are empty\n",
    "        print(f\"⚠️ Warning: Empty embeddings for batch {i+1}, skipping...\")\n",
    "        continue\n",
    "\n",
    "    embeddings_list.append(batch_embeddings)\n",
    "\n",
    "# Convert list of arrays to a single NumPy array\n",
    "if embeddings_list:  # Ensure there are valid embeddings before stacking\n",
    "    embeddings = np.vstack(embeddings_list)\n",
    "else:\n",
    "    print(\"❌ Error: No embeddings generated!\")\n",
    "    exit()\n",
    "\n",
    "# Save FAISS index\n",
    "dimension = embeddings.shape[1]\n",
    "index = faiss.IndexFlatL2(dimension)\n",
    "index.add(embeddings)\n",
    "faiss.write_index(index, \"faiss_index.bin\")\n",
    "np.save(\"embeddings.npy\", embeddings)\n",
    "\n",
    "print(\"🎉✅ Preprocessing complete! Data cleaned, embeddings generated, and FAISS index saved.\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "0a1109a8-e231-4bdd-9e0b-ae987bae4ed9",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "✅ Processed dataset loaded! Shape: (1000, 33)\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>hotel</th>\n",
       "      <th>is_canceled</th>\n",
       "      <th>lead_time</th>\n",
       "      <th>arrival_date_year</th>\n",
       "      <th>arrival_date_month</th>\n",
       "      <th>arrival_date_week_number</th>\n",
       "      <th>arrival_date_day_of_month</th>\n",
       "      <th>stays_in_weekend_nights</th>\n",
       "      <th>stays_in_week_nights</th>\n",
       "      <th>adults</th>\n",
       "      <th>...</th>\n",
       "      <th>agent</th>\n",
       "      <th>company</th>\n",
       "      <th>days_in_waiting_list</th>\n",
       "      <th>customer_type</th>\n",
       "      <th>adr</th>\n",
       "      <th>required_car_parking_spaces</th>\n",
       "      <th>total_of_special_requests</th>\n",
       "      <th>reservation_status</th>\n",
       "      <th>reservation_status_date</th>\n",
       "      <th>revenue</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Resort Hotel</td>\n",
       "      <td>0</td>\n",
       "      <td>203</td>\n",
       "      <td>2016</td>\n",
       "      <td>December</td>\n",
       "      <td>49</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>5</td>\n",
       "      <td>2</td>\n",
       "      <td>...</td>\n",
       "      <td>250.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0</td>\n",
       "      <td>Transient</td>\n",
       "      <td>66.8</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>Check-Out</td>\n",
       "      <td>2016-12-09</td>\n",
       "      <td>467.6</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>City Hotel</td>\n",
       "      <td>1</td>\n",
       "      <td>82</td>\n",
       "      <td>2015</td>\n",
       "      <td>July</td>\n",
       "      <td>29</td>\n",
       "      <td>16</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>2</td>\n",
       "      <td>...</td>\n",
       "      <td>9.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0</td>\n",
       "      <td>Transient</td>\n",
       "      <td>76.5</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>Canceled</td>\n",
       "      <td>2015-07-16</td>\n",
       "      <td>229.5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>City Hotel</td>\n",
       "      <td>0</td>\n",
       "      <td>25</td>\n",
       "      <td>2016</td>\n",
       "      <td>December</td>\n",
       "      <td>53</td>\n",
       "      <td>27</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>3</td>\n",
       "      <td>...</td>\n",
       "      <td>220.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0</td>\n",
       "      <td>Transient-Party</td>\n",
       "      <td>60.0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>Check-Out</td>\n",
       "      <td>2016-12-30</td>\n",
       "      <td>180.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>City Hotel</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>2016</td>\n",
       "      <td>March</td>\n",
       "      <td>11</td>\n",
       "      <td>9</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>...</td>\n",
       "      <td>9.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0</td>\n",
       "      <td>Transient-Party</td>\n",
       "      <td>95.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>Check-Out</td>\n",
       "      <td>2016-03-10</td>\n",
       "      <td>95.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>City Hotel</td>\n",
       "      <td>0</td>\n",
       "      <td>70</td>\n",
       "      <td>2017</td>\n",
       "      <td>April</td>\n",
       "      <td>16</td>\n",
       "      <td>16</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>...</td>\n",
       "      <td>9.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0</td>\n",
       "      <td>Transient</td>\n",
       "      <td>108.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>Check-Out</td>\n",
       "      <td>2017-04-20</td>\n",
       "      <td>432.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5 rows × 33 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "          hotel  is_canceled  lead_time  arrival_date_year arrival_date_month  \\\n",
       "0  Resort Hotel            0        203               2016           December   \n",
       "1    City Hotel            1         82               2015               July   \n",
       "2    City Hotel            0         25               2016           December   \n",
       "3    City Hotel            0          1               2016              March   \n",
       "4    City Hotel            0         70               2017              April   \n",
       "\n",
       "   arrival_date_week_number  arrival_date_day_of_month  \\\n",
       "0                        49                          2   \n",
       "1                        29                         16   \n",
       "2                        53                         27   \n",
       "3                        11                          9   \n",
       "4                        16                         16   \n",
       "\n",
       "   stays_in_weekend_nights  stays_in_week_nights  adults  ...  agent  company  \\\n",
       "0                        2                     5       2  ...  250.0      NaN   \n",
       "1                        0                     3       2  ...    9.0      NaN   \n",
       "2                        0                     3       3  ...  220.0      NaN   \n",
       "3                        0                     1       1  ...    9.0      NaN   \n",
       "4                        2                     2       2  ...    9.0      NaN   \n",
       "\n",
       "  days_in_waiting_list    customer_type    adr required_car_parking_spaces  \\\n",
       "0                    0        Transient   66.8                           0   \n",
       "1                    0        Transient   76.5                           0   \n",
       "2                    0  Transient-Party   60.0                           0   \n",
       "3                    0  Transient-Party   95.0                           0   \n",
       "4                    0        Transient  108.0                           0   \n",
       "\n",
       "   total_of_special_requests  reservation_status  reservation_status_date  \\\n",
       "0                          0           Check-Out               2016-12-09   \n",
       "1                          0            Canceled               2015-07-16   \n",
       "2                          1           Check-Out               2016-12-30   \n",
       "3                          0           Check-Out               2016-03-10   \n",
       "4                          0           Check-Out               2017-04-20   \n",
       "\n",
       "  revenue  \n",
       "0   467.6  \n",
       "1   229.5  \n",
       "2   180.0  \n",
       "3    95.0  \n",
       "4   432.0  \n",
       "\n",
       "[5 rows x 33 columns]"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import pandas as pd\n",
    "\n",
    "df = pd.read_csv(\"data/processed_data.csv\")\n",
    "print(f\"✅ Processed dataset loaded! Shape: {df.shape}\")\n",
    "df.head()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "d4950dbd-9a76-425c-b309-aede330aae8a",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "✅ FAISS index loaded! Number of vectors: 1000\n"
     ]
    }
   ],
   "source": [
    "import faiss\n",
    "import numpy as np\n",
    "\n",
    "index = faiss.read_index(\"faiss_index.bin\")\n",
    "print(f\"✅ FAISS index loaded! Number of vectors: {index.ntotal}\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "e0b1b9cd-c9e9-49fe-bf1c-eb4c0be1d490",
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "%run database_setup.py"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "d21603a8-97af-44f8-b36e-8a45440a3f2f",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "✅ Database file found: hotel_analytics.db\n"
     ]
    }
   ],
   "source": [
    "import os\n",
    "\n",
    "if os.path.exists(\"hotel_analytics.db\"):\n",
    "    print(\"✅ Database file found: hotel_analytics.db\")\n",
    "else:\n",
    "    print(\"❌ Database not created! Something went wrong.\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "7e080021-da07-43aa-9559-1a370552fb49",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "📊 total_revenue: 343853.12\n",
      "📊 cancellation_rate: 38.5\n",
      "📊 top_countries: {'PRT': 411, 'GBR': 104, 'ESP': 80, 'FRA': 78, 'DEU': 69, 'ITA': 31, 'IRL': 29, 'USA': 20, 'BEL': 18, 'BRA': 16}\n",
      "📊 lead_time_distribution: {'count': 1000.0, 'mean': 106.232, 'std': 106.68598818999652, 'min': 0.0, '25%': 20.0, '50%': 73.0, '75%': 164.25, 'max': 594.0}\n"
     ]
    }
   ],
   "source": [
    "import sqlite3\n",
    "\n",
    "conn = sqlite3.connect(\"hotel_analytics.db\")\n",
    "cursor = conn.cursor()\n",
    "\n",
    "# Fetch all stored analytics\n",
    "cursor.execute(\"SELECT * FROM analytics\")\n",
    "results = cursor.fetchall()\n",
    "\n",
    "if results:\n",
    "    for row in results:\n",
    "        print(f\"📊 {row[0]}: {row[1]}\")\n",
    "else:\n",
    "    print(\"❌ No analytics data found in the database!\")\n",
    "\n",
    "conn.close()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4cdcf26b-bc98-459c-b776-32204ac5d3fc",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
