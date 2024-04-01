## Integration with Vast.AI

The script can be executed remotely.

The https://vast.ai is a global GPU market, here you can rent a machine with powerful GPU and run magnetic fields simulation remotely.

At first you must have an account on https://vast.ai and enough account balance.

Vast.AI provides command-line client tool and API integration.

Here are few steps you need to do first to to be able to rent an instance and start the script remotely.



###### SSH key

Add public part of your SSH key to the Vast.AI account: https://cloud.vast.ai/account/



###### Install Vast.AI client locally

        make vastai_install



###### API key

Now you need to authorise yourself using Vast.AI API key.

Go to https://cloud.vast.ai/account/ and copy the API key shown in your account.

Run following command locally:

        make vastai_login API_KEY=abcdef123456...



###### Rent an instance

Now it is time to rent hardware from the Vast.AI market. You can do that via web-site here: https://cloud.vast.ai/create/

Another option is to do that via command-line. To print all of the available offers, run following command locally:

        make vastai_offers


Now you need to decide which machine you are going to use and copy the ID of the offer - this is a value from the first column in the table.

Here is a short command to rent a machine via command line - this will also automatically install all required libraries for you:

        make vastai_instance_create ID=123456


A new `current_contract_id.txt` file is created locally which holds your contract ID with Vast.AI. This is a different ID, not the offer ID you used in a previous step. The contract ID is required to manipulate your running instance.

It will take some time to launch your instance - it will not be accesible right away.

To destroy the instance you can simply run bellow command - do not forget to do that after you finished with script simulations (because you still have to pay for it, even when you are not using it):

        make vastai_instance_destroy



###### SSH to the Vast.AI instance

This is a quick way to get into the running instance via SSH:

        make vastai_ssh



###### Deploy source code into the instance

Following command will clone that repository into the running instance:

        make vastai_getcode



###### Run magnetic fields simulation remotely

Now, after you rented and prepared a remote instance, you can finally start the script.

Bellow command will , clone this repository into the instance and execute the simulation:

        make vastai_run SIMULATION_NUMBER=1


Above command will also automatically generate report files inside of the instance.

To download results to your host simply run the following command:

        make vastai_download


Reminder, do not forget to stop and destroy your running instance after you done with simulations, run `make vastai_instance_destroy`.
